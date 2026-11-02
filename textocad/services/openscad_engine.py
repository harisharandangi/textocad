"""3D Printing engine: LLM-authored OpenSCAD, compiled by the OpenSCAD binary."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from textocad.libs import fetch_bosl2, lib_root
from textocad.services.codegen_sandbox import ExecResult

LIB_ROOT = lib_root()
MAX_SOURCE_CHARS = 60_000
MAX_PARTS = 24
MAX_FN = 192
PART_TIMEOUT_S = int(os.getenv("TEXTOCAD_OPENSCAD_PART_TIMEOUT", "90"))
MEM_MB = int(os.getenv("TEXTOCAD_OPENSCAD_MEM_MB", "2048"))
PARALLEL = max(1, int(os.getenv("TEXTOCAD_OPENSCAD_PARALLEL", "2")))

_PART_MODULE_RE = re.compile(r"\bmodule\s+part_([A-Za-z0-9_]+)\s*\(\s*\)")
_PART_TAG_RE = re.compile(
    r"//\s*@part\s+([A-Za-z0-9_]+)\s*(#[0-9a-fA-F]{6}|[A-Za-z]+)?[ \t]*([^\n]*)"
)
_TITLE_RE = re.compile(r"//\s*title\s*:\s*(.+)", re.IGNORECASE)
_LIB_RE = re.compile(r"\b(include|use)\s*<([^>]*)>")


@dataclass
class OpenScadBinary:
    path: str
    version: str
    manifold_flag: list[str]
    binary_stl: bool


_BIN_LOCK = threading.Lock()
_BIN: OpenScadBinary | None = None


def find_openscad() -> OpenScadBinary | None:
    """The best OpenSCAD on this machine: $OPENSCAD_BIN, then the nightly (which has the fast Manifold backend),."""
    global _BIN
    with _BIN_LOCK:
        if _BIN is not None:
            return _BIN
        candidates = [
            os.getenv("OPENSCAD_BIN", ""),
            shutil.which("openscad-nightly") or "",
            shutil.which("openscad") or "",
            r"C:\Program Files\OpenSCAD (Nightly)\openscad.com",
            r"C:\Program Files\OpenSCAD\openscad.com",
        ]
        for cand in candidates:
            if not cand or not Path(cand).exists():
                continue
            try:
                ver = subprocess.run(
                    [cand, "--version"], capture_output=True, text=True, timeout=20
                )
                text = (ver.stdout + ver.stderr).strip()
                helptext = subprocess.run(
                    [cand, "--help"], capture_output=True, text=True, timeout=20
                )
                help_all = helptext.stdout + helptext.stderr
            except Exception:
                continue
            m = re.search(r"version\s+([0-9][0-9.]*)", text)
            version = m.group(1) if m else text[:40]
            if "--backend" in help_all:
                flag = ["--backend=manifold"]
            elif "manifold" in help_all:
                flag = ["--enable=manifold"]
            else:
                flag = []
            year = int(version.split(".")[0]) if version[:4].isdigit() else 0
            _BIN = OpenScadBinary(
                path=cand, version=version, manifold_flag=flag, binary_stl=year >= 2022
            )
            return _BIN
        return None


def engine_status() -> dict[str, Any]:
    b = find_openscad()
    return {
        "available": b is not None,
        "version": b.version if b else None,
        "manifold": bool(b and b.manifold_flag),
        "libraries": sorted(p.name for p in LIB_ROOT.iterdir()) if LIB_ROOT.exists() else [],
    }


@dataclass
class ScadPart:
    name: str
    color: str | None = None
    label: str = ""


@dataclass
class ScadModel:
    source: str
    title: str
    params: list[dict[str, Any]] = field(default_factory=list)
    parts: list[ScadPart] = field(default_factory=list)


def _split_top_level(source: str) -> list[tuple[str, str]]:
    """Split OpenSCAD source into top-level pieces: ('comment'|'ws'|'stmt', text)."""
    out: list[tuple[str, str]] = []
    i, n = 0, len(source)
    while i < n:
        ch = source[i]
        if ch.isspace():
            j = i
            while j < n and source[j].isspace():
                j += 1
            out.append(("ws", source[i:j]))
            i = j
            continue
        if source.startswith("//", i):
            j = source.find("\n", i)
            j = n if j < 0 else j
            out.append(("comment", source[i:j]))
            i = j
            continue
        if source.startswith("/*", i):
            j = source.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(("comment", source[i:j]))
            i = j
            continue
        if re.match(r"(include|use)\s*<", source[i:i + 12]):
            j = source.find(">", i)
            j = n if j < 0 else j + 1
            out.append(("stmt", source[i:j]))
            i = j
            continue
        depth = 0
        j = i
        while j < n:
            c = source[j]
            if source.startswith("//", j):
                k = source.find("\n", j)
                j = n if k < 0 else k
                continue
            if source.startswith("/*", j):
                k = source.find("*/", j + 2)
                j = n if k < 0 else k + 2
                continue
            if c == '"':
                j += 1
                while j < n and source[j] != '"':
                    j += 2 if source[j] == "\\" else 1
                j += 1
                continue
            if c in "({[":
                depth += 1
            elif c in ")}]":
                depth -= 1
                if c == "}" and depth <= 0:
                    j += 1
                    break
            elif c == ";" and depth <= 0:
                j += 1
                break
            j += 1
        out.append(("stmt", source[i:j]))
        i = j
    return out


def _is_instantiation(stmt: str) -> bool:
    s = stmt.strip()
    if not s:
        return False
    if re.match(r"(module|function)\b", s) or re.match(r"(include|use)\s*<", s):
        return False
    if re.match(r"\$?[A-Za-z_][A-Za-z0-9_]*\s*=", s):
        return False
    return True


PART_GATE = "_textocad_part"
_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_MODULE_DEF_RE = re.compile(r"^\s*module\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.S)
_PART_DEF_RE = re.compile(r"^(\s*module\s+part_([A-Za-z0-9_]+)\s*\(\s*\))(.*)$", re.S)


def _code_only(text: str) -> str:
    """Text with comments and string literals blanked, for call scanning."""
    text = re.sub(r"//[^\n]*|/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', text)


def isolate_parts(source: str) -> tuple[str, list[str], bool]:
    """Make one script render any single part, WITH the placement the author gave it."""
    pieces = _split_top_level(source)
    bodies: dict[str, str] = {}
    for kind, text in pieces:
        m = _MODULE_DEF_RE.match(text) if kind == "stmt" else None
        if m:
            bodies[m.group(1)] = _code_only(text[m.end():])
    names = [n[len("part_"):] for n in bodies if n.startswith("part_")]


    draws_part: set[str] = {f"part_{n}" for n in names}
    changed = True
    while changed:
        changed = False
        for mod, body in bodies.items():
            if mod not in draws_part and any(c in draws_part for c in _CALL_RE.findall(body)):
                draws_part.add(mod)
                changed = True

    out: list[str] = [f'{PART_GATE} = "";\n']
    reached: set[str] = set()
    has_other = False
    for kind, text in pieces:
        if kind != "stmt":
            out.append(text)
            continue
        m = _PART_DEF_RE.match(text)
        if m:
            head, name, body = m.group(1), m.group(2), m.group(3).strip()
            out.append(f'{head} {{ if ({PART_GATE} == "" || {PART_GATE} == "{name}") {body} }}')
            continue
        if _is_instantiation(text):
            calls = set(_CALL_RE.findall(_code_only(text)))

            frontier, seen = list(calls), set()
            while frontier:
                c = frontier.pop()
                if c in seen:
                    continue
                seen.add(c)
                frontier.extend(_CALL_RE.findall(bodies.get(c, "")))
            reached |= {c[len("part_"):] for c in seen if c.startswith("part_")}
            if calls & draws_part:
                out.append(text)
            else:
                has_other = True
                out.append(f'if ({PART_GATE} == "" || {PART_GATE} == "other") {{ {text.strip()} }}')
            continue
        out.append(text)
    for name in names:
        if name not in reached:
            out.append(f"\npart_{name}();")
    return "".join(out) + "\n", names, has_other


def _parse_number(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def parse_params(source: str) -> list[dict[str, Any]]:
    """Customizer parameters -> param_schema."""
    params: list[dict[str, Any]] = []
    group = "Parameters"
    seen: set[str] = set()
    for line in source.splitlines():
        stripped = line.strip()
        g = re.match(r"/\*\s*\[([^\]]+)\]\s*\*/", stripped)
        if g:
            group = g.group(1).strip()
            continue
        if re.match(r"(module|function)\b", stripped):
            break
        if group.lower() == "hidden":
            continue
        m = re.match(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?[0-9.]+|true|false)\s*;\s*(?://\s*(.*))?$",
            stripped,
        )
        if not m or m.group(1) in seen:
            continue
        key, raw_value, comment = m.group(1), m.group(2), (m.group(3) or "").strip()
        seen.add(key)
        is_bool = raw_value in ("true", "false")
        value = 1.0 if raw_value == "true" else 0.0 if raw_value == "false" else _parse_number(raw_value)
        if value is None:
            continue
        lo = hi = step = None
        label = ""
        rng = re.match(r"\[([^\]]*)\]\s*(.*)", comment)
        if rng:
            nums = [_parse_number(x) for x in rng.group(1).split(":")]
            nums = [x for x in nums if x is not None]
            if len(nums) == 3:
                lo, step, hi = nums
            elif len(nums) == 2:
                lo, hi = nums
            elif len(nums) == 1:
                hi = nums[0]
                lo = 0.0
            label = rng.group(2).strip()
        else:
            label = comment
        unit = ""
        um = re.search(r"\((mm|deg|°|degrees|%)\)", label)
        if um:
            unit = "deg" if um.group(1) in ("deg", "°", "degrees") else um.group(1)
            label = label.replace(um.group(0), "").strip()
        if is_bool:
            lo, hi, step = 0.0, 1.0, 1.0
        if lo is None:
            lo = 0.0 if value >= 0 else value * 2
        if hi is None:
            hi = max(value * 2, value + 10, 1.0)
        if step is None:
            step = 1.0 if float(value).is_integer() and (hi - lo) >= 10 else 0.1
        params.append(
            {
                "key": key,
                "label": label or key.replace("_", " ").capitalize(),
                "type": "boolean" if is_bool else "number",
                "value": value,
                "min": lo,
                "max": hi,
                "step": step,
                "unit": unit or ("mm" if not is_bool else ""),
                "group": group,
            }
        )
    return params


def parse_model(source: str, *, keep_placement: bool = True) -> ScadModel:
    """``keep_placement=False`` drops top-level geometry calls first: robot links must stay in their own frames,."""
    if not keep_placement:
        source = "".join(
            text for kind, text in _split_top_level(source)
            if not (kind == "stmt" and _is_instantiation(text))
        )
    title_m = _TITLE_RE.search(source)
    title = title_m.group(1).strip()[:80] if title_m else "3D print model"
    tags = {m.group(1): (m.group(2), m.group(3).strip()) for m in _PART_TAG_RE.finditer(source)}
    if _PART_MODULE_RE.search(source):
        kept, names, has_other = isolate_parts(source)
        parts = []
        for name in names[:MAX_PARTS]:
            color, label = tags.get(name, (None, ""))
            parts.append(ScadPart(name=name, color=color, label=label))
        if has_other:
            parts.append(ScadPart(name="other", label="Other"))
    else:
        kept = source
        parts = [ScadPart(name="body")]
    return ScadModel(source=kept, title=title, params=parse_params(source), parts=parts)


def validate_scad(source: str) -> list[str]:
    errors: list[str] = []
    if not source or not source.strip():
        return ["script is empty"]
    if len(source) > MAX_SOURCE_CHARS:
        errors.append(f"script is longer than {MAX_SOURCE_CHARS} characters")
    code = re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.DOTALL)
    for kind, lib in _LIB_RE.findall(code):
        lib = lib.strip()
        if not lib.startswith("BOSL2/") or ".." in lib or lib.startswith("/"):
            errors.append(f"{kind} <{lib}> is not allowed (only BOSL2 library files)")
    if re.search(r"\bimport\s*\(", code):
        errors.append("import() is not allowed (no external files)")
    if re.search(r"\bsurface\s*\(", code):
        errors.append("surface() is not allowed (no external files)")
    for m in re.finditer(r"\$fn\s*=\s*([0-9.]+)", code):
        if float(m.group(1)) > MAX_FN:
            errors.append(f"$fn={m.group(1)} is too high (max {MAX_FN})")
    if not re.search(r"\b(module|cube|sphere|cylinder|polyhedron|linear_extrude|rotate_extrude|cuboid|cyl)\b", code):
        errors.append("script draws no geometry")
    return errors


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return '"' + str(value).replace('"', "") + '"'
    return str(int(f)) if f.is_integer() else repr(f)


def define_args(params: dict[str, Any], schema: list[dict[str, Any]] | None = None) -> list[str]:
    """-D overrides for slider values. Only identifiers; booleans from the schema go back to true/false."""
    kinds = {p.get("key"): p.get("type") for p in (schema or [])}
    args: list[str] = []
    for key, value in (params or {}).items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key)):
            continue
        if kinds.get(key) == "boolean":
            value = bool(float(value)) if not isinstance(value, bool) else value
        args += ["-D", f"{key}={_format_value(value)}"]
    return args


def _limits() -> Any:
    if os.name != "posix":
        return None

    def apply() -> None:
        try:
            import resource

            mem = MEM_MB * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        except Exception:
            pass
        try:
            with open("/proc/self/oom_score_adj", "w") as fh:
                fh.write("1000")
        except Exception:
            pass

    return apply


def _run_part(binary: OpenScadBinary, work: Path, part: ScadPart, defines: list[str]) -> tuple[str, str | None]:


    entry = work / "model.scad"
    if part.name != "body" or _PART_MODULE_RE.search(entry.read_text("utf-8")):
        defines = defines + ["-D", f'{PART_GATE}="{part.name}"']
    out = work / "parts" / f"{part.name}.stl"
    cmd = [binary.path, "-o", str(out)]
    if binary.binary_stl:
        cmd += ["--export-format", "binstl"]
    cmd += binary.manifold_flag + defines + [str(entry)]
    env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "HOME", "LANG") if k in os.environ}
    env["OPENSCADPATH"] = str(LIB_ROOT)
    try:
        proc = subprocess.run(
            cmd, cwd=str(work), capture_output=True, text=True, timeout=PART_TIMEOUT_S,
            env=env, preexec_fn=_limits(),
        )
    except subprocess.TimeoutExpired:
        return part.name, f"part '{part.name}' took longer than {PART_TIMEOUT_S}s to compile (simplify it)"
    log = (proc.stderr or "") + (proc.stdout or "")
    if proc.returncode != 0 or not out.exists() or out.stat().st_size < 84:
        errs = [l for l in log.splitlines() if "ERROR" in l or "WARNING: Ignoring unknown" in l]
        detail = "\n".join(errs[-6:]) or log.strip()[-600:]
        if "Current top level object is empty" in log or (out.exists() and out.stat().st_size < 84):
            detail = f"part_{part.name}() produced no geometry. " + detail
        return part.name, f"OpenSCAD failed on part '{part.name}': {detail}"
    return part.name, None


def compile_model(
    source: str,
    params: dict[str, Any] | None,
    build_dir: str | Path,
    *,
    schema: list[dict[str, Any]] | None = None,
    keep_placement: bool = True,
) -> ExecResult:
    """Compile every part, then bundle."""
    errors = validate_scad(source)
    if errors:
        return ExecResult(ok=False, error="OpenSCAD validation failed: " + "; ".join(errors))
    binary = find_openscad()
    if binary is None:
        return ExecResult(ok=False, error="OpenSCAD is not installed or not on PATH (see README)")
    if "BOSL2/" in source:
        try:
            fetch_bosl2()
        except Exception as exc:
            return ExecResult(ok=False, error=f"could not download the BOSL2 library: {exc}")
    model = parse_model(source, keep_placement=keep_placement)
    bdir = Path(build_dir)
    bdir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="scad_", dir=str(bdir)))
    (work / "parts").mkdir()
    (work / "model.scad").write_text(model.source, encoding="utf-8")
    merged = {p["key"]: p["value"] for p in (schema or model.params)}
    merged.update(params or {})
    defines = define_args(merged, schema or model.params)

    with ThreadPoolExecutor(max_workers=min(PARALLEL, len(model.parts))) as pool:
        results = list(pool.map(lambda p: _run_part(binary, work, p, defines), model.parts))
    failures = [err for _, err in results if err]
    if failures:
        shutil.rmtree(work, ignore_errors=True)
        return ExecResult(ok=False, error="\n".join(failures[:4]))

    from textocad.services.geometry_isolation import IsolationError, run_isolated
    from textocad.services.part_export import PARTS_DIR, write_bundle

    pdir = bdir / PARTS_DIR
    if pdir.exists():
        shutil.rmtree(pdir, ignore_errors=True)
    pdir.mkdir(parents=True)
    listed = []
    for part in model.parts:
        dst = pdir / f"{part.name}.stl"
        shutil.move(str(work / "parts" / f"{part.name}.stl"), dst)
        listed.append({"name": part.name, "color": part.color, "label": part.label, "stl_path": str(dst)})
    shutil.rmtree(work, ignore_errors=True)
    try:
        bundle = run_isolated(write_bundle, listed, str(pdir), timeout_s=180)
    except (IsolationError, ValueError) as exc:
        return ExecResult(ok=False, error=f"could not assemble the parts: {exc}")
    scad_path = bdir / "result.scad"
    scad_path.write_text(source, encoding="utf-8")
    final: dict[str, str] = {}
    for key, name in (("stl_path", "result.stl"), ("glb_path", "result.glb"), ("threemf_path", "result.3mf")):
        dst = bdir / name
        shutil.move(bundle[key], dst)
        final[key] = str(dst)
    return ExecResult(
        ok=True,
        stl_path=final["stl_path"],
        glb_path=final["glb_path"],
        threemf_path=final["threemf_path"],
        source_path=str(scad_path),
        bbox=tuple(bundle["bbox"]),
        volume=bundle["volume"],
        parts=bundle["parts"],
    )
