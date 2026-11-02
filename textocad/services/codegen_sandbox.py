from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path


_ALLOWED_IMPORTS_PRECISION = {"cadquery", "cq_gears", "math"}
_ALLOWED_IMPORTS_ORGANIC = {"trimesh", "manifold3d", "numpy", "math"}

_BANNED_NAMES = {
    "eval", "exec", "compile", "open", "input", "__import__", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "hasattr", "breakpoint", "memoryview",
    "exit", "quit", "help", "dir",
}

_BANNED_ATTR_PREFIXES = ("__",)
_BANNED_ATTRS = {
    "__class__", "__bases__", "__subclasses__", "__mro__", "__globals__",
    "__builtins__", "__dict__", "__code__", "__closure__", "__import__",
    "__getattribute__", "__subclasshook__", "__reduce__", "__reduce_ex__",
    "os", "sys", "subprocess", "shutil", "socket", "ctypes", "ctypeslib", "pathlib",
    "io", "tempfile", "importlib", "builtins", "pickle", "marshal", "environ",
    "getenv", "system", "popen", "spawn", "remove", "unlink", "rmtree", "chdir",
    "load", "loads", "save", "savez", "savez_compressed", "savetxt", "loadtxt",
    "genfromtxt", "fromfile", "tofile", "fromregex", "memmap", "open_memmap",
    "dump", "DataSource", "load_library", "lib", "f2py", "distutils", "testing",
    "export", "exporters", "importers", "importStep", "importBrep", "importStl",
    "importDXF", "exportStep", "exportStl", "exportBrep", "exportBin", "exportSvg",
    "export_mesh", "load_mesh", "load_path", "load_remote", "load_scene",
    "resolvers", "exchange", "interfaces", "viewer", "show",
}

_MAX_AST_NODES = 6000
_MAX_LOOP_NODES = 60


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)


def allowed_imports(mode: str) -> set[str]:
    return _ALLOWED_IMPORTS_ORGANIC if mode == "organic" else _ALLOWED_IMPORTS_PRECISION


def validate_script(source: str, mode: str = "precision") -> ValidationResult:
    """Static AST allowlist."""
    errors: list[str] = []
    allowed = allowed_imports(mode)

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ValidationResult(False, [f"syntax error: {exc}"])

    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_AST_NODES:
        errors.append(f"script too large ({len(nodes)} AST nodes > {_MAX_AST_NODES})")

    loop_count = sum(isinstance(n, (ast.For, ast.While)) for n in nodes)
    if loop_count > _MAX_LOOP_NODES:
        errors.append(f"too many loops ({loop_count} > {_MAX_LOOP_NODES})")

    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in allowed:
                    errors.append(f"import of '{alias.name}' not allowed (allowed: {sorted(allowed)})")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in allowed:
                errors.append(f"from-import of '{node.module}' not allowed (allowed: {sorted(allowed)})")
            for alias in node.names:
                if alias.name == "*" or alias.name in _BANNED_ATTRS or alias.name.startswith("_"):
                    errors.append(f"import of '{alias.name}' from '{node.module}' not allowed")

    for node in nodes:
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            errors.append(f"use of banned name '{node.id}'")
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr in _BANNED_ATTRS or attr.startswith(_BANNED_ATTR_PREFIXES):
                errors.append(f"access to forbidden attribute '{attr}'")

    top = tree.body
    has_build = any(
        isinstance(n, ast.FunctionDef) and n.name == "build"
        and len(n.args.args) >= 1
        for n in top
    )
    if not has_build:
        errors.append("script must define a top-level function `build(params)`")

    for n in top:
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import,
                              ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.Expr)):
            errors.append(f"unexpected top-level statement {type(n).__name__} "
                          f"(only defs/imports allowed at module level)")
        if isinstance(n, ast.Expr) and not isinstance(n.value, ast.Constant):
            errors.append("no executable code allowed at module top level (only defs)")

    return ValidationResult(ok=len(errors) == 0, errors=errors)


_SAFE_BUILTINS = [
    "abs", "min", "max", "round", "sum", "len", "range", "enumerate", "zip",
    "map", "filter", "sorted", "reversed", "float", "int", "bool", "str",
    "list", "tuple", "dict", "set", "frozenset", "all", "any", "divmod", "pow",
    "True", "False", "None", "print", "isinstance", "issubclass",
    "Exception", "BaseException", "ValueError", "TypeError", "RuntimeError",
    "ArithmeticError", "ZeroDivisionError", "OverflowError", "FloatingPointError",
    "IndexError", "KeyError", "AttributeError", "NotImplementedError",
    "StopIteration", "AssertionError", "Warning",
]


@dataclass
class ExecResult:
    ok: bool
    error: str = ""
    step_path: str | None = None
    stl_path: str | None = None
    glb_path: str | None = None
    bbox: tuple[float, float, float] | None = None
    volume: float | None = None
    stdout: str = ""
    parts: list = field(default_factory=list)
    threemf_path: str | None = None
    source_path: str | None = None


def _runner_source(mode: str) -> str:
    """The trusted harness that runs INSIDE the subprocess."""
    safe = ", ".join(repr(b) for b in _SAFE_BUILTINS)
    common = f'''
import json, sys, builtins as _b
try:
    import resource  # POSIX only
except Exception:
    resource = None

try:
    with open("/proc/self/oom_score_adj", "w") as _adj:
        _adj.write("1000")
except Exception:
    pass

def _apply_limits(mem_bytes, cpu_secs):
    if resource is None:
        return
    try:
        if mem_bytes and mem_bytes > 0:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        if cpu_secs and cpu_secs > 0:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_secs, cpu_secs))
    except Exception:
        pass

_cfg = json.loads(sys.stdin.read())
script = _cfg["script"]; params = _cfg["params"]; build_dir = _cfg["build_dir"]
if _cfg.get("harness_root"):
    sys.path.insert(0, _cfg["harness_root"])
mem_mb = int(_cfg.get("mem_mb", 1024)); cpu_s = int(_cfg.get("cpu_s", 60))
_apply_limits(mem_mb * 1024 * 1024, cpu_s)

_allowed_builtins = {{name: getattr(_b, name) for name in [{safe}] if hasattr(_b, name)}}
_ROOTS = frozenset(_cfg.get("allowed_roots") or ())
_BLOCKED = frozenset(_cfg.get("blocked_attrs") or ())
'''
    guard = '''
import types as _types, socket as _socket

def _no_network(*_a, **_k):
    raise OSError("network access is disabled in the build sandbox")

for _name in ("socket", "create_connection", "getaddrinfo", "socketpair", "fromfd"):
    setattr(_socket, _name, _no_network)


_REAL = {}


class _Module:
    """Read-only view of an allowed module. Submodules of the same package pass
    through (wrapped); anything else that is a module, and file/process
    helpers, are refused, so `trimesh.util.os` or `np.ctypeslib` lead nowhere."""

    __slots__ = ("__weakref__",)

    def __init__(self, module):
        _REAL[id(self)] = module

    def __getattr__(self, name):
        if name in _BLOCKED:
            raise AttributeError(name + " is not available in build scripts")
        value = getattr(_REAL[id(self)], name)
        if isinstance(value, _types.ModuleType):
            if value.__name__.split(".")[0] not in _ROOTS:
                raise AttributeError(name + " is not available in build scripts")
            return _wrap(value)
        return value

    def __setattr__(self, name, value):
        raise AttributeError("modules are read-only in build scripts")


_WRAPPED = {}


def _wrap(module):
    proxy = _WRAPPED.get(module.__name__)
    if proxy is None:
        proxy = _WRAPPED[module.__name__] = _Module(module)
    return proxy


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in _ROOTS:
        raise ImportError("import of " + repr(name) + " is not allowed")
    module = _b.__import__(name, globals, locals, fromlist, level)
    return _wrap(module)


_allowed_builtins["__import__"] = _guarded_import
'''
    if mode == "organic":
        body = '''
import os, trimesh, numpy as np  # noqa
def _result_out(obj):
    # Accept a trimesh.Trimesh or something convertible
    mesh = obj
    if hasattr(obj, "to_mesh"):
        mesh = obj.to_mesh()
    if not isinstance(mesh, trimesh.Trimesh):
        try:
            mesh = trimesh.Trimesh(vertices=obj.vertices, faces=obj.faces)
        except Exception:
            raise ValueError("organic build() must return a trimesh.Trimesh (or convertible)")
    if mesh.is_empty or len(mesh.faces) == 0:
        raise ValueError("organic result is empty (no faces)")
    # PULSE CHECK (organic): volume>0 + at least a closed-ish mesh. Mesh is the kernel here.
    vol = float(abs(mesh.volume))
    if vol <= 1e-9:
        raise ValueError("organic result has zero volume")
    stl = os.path.join(build_dir, "result.stl"); mesh.export(stl)
    glb = os.path.join(build_dir, "result.glb"); mesh.export(glb)
    ext = mesh.extents
    return {"ok": True, "stl_path": stl, "glb_path": glb,
            "bbox": [float(ext[0]), float(ext[1]), float(ext[2])], "volume": vol}
'''
    else:
        body = '''
import os, cadquery as cq  # noqa
from textocad.services.part_export import normalize_color, safe_part_name, write_bundle, PARTS_DIR

def _solid_of(wp, name):
    try:
        val = wp.val() if hasattr(wp, "val") else wp
    except Exception:
        raise ValueError(f"part '{name}': build() must return cadquery Workplanes/Solids")
    # A Workplane holding several solids is one part: fuse into a compound.
    try:
        if hasattr(wp, "vals") and len(wp.vals()) > 1:
            val = cq.Compound.makeCompound([v for v in wp.vals() if hasattr(v, "isValid")])
    except Exception:
        pass
    # PULSE CHECK (precision): B-rep validity is AUTHORITATIVE (not mesh watertightness)
    try:
        valid = bool(val.isValid())
    except Exception as exc:
        raise ValueError(f"part '{name}': could not validate solid: " + str(exc))
    if not valid:
        raise ValueError(f"part '{name}' is not a valid solid (B-rep invalid/self-intersecting)")
    try:
        vol = float(val.Volume())
    except Exception:
        vol = 0.0
    if vol <= 1e-9:
        raise ValueError(f"part '{name}' is empty (zero volume)")
    return val

def _parts_of(obj):
    # build() may return ONE solid, a {name: solid} dict, a {name: (solid, colour)}
    # dict, a list of (name, solid[, colour]) tuples, or a cq.Assembly.
    raw = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (tuple, list)) and len(v) >= 2:
                raw.append((k, v[0], v[1]))
            else:
                raw.append((k, v, None))
    elif isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], (list, tuple)):
        for item in obj:
            raw.append((item[0], item[1], item[2] if len(item) > 2 else None))
    elif isinstance(obj, cq.Assembly):
        for child in obj.traverse():
            node = child[1]
            if node.obj is None:
                continue
            shape = node.obj.val() if hasattr(node.obj, "val") else node.obj
            shape = shape.moved(node.loc) if hasattr(shape, "moved") else shape
            col = node.color.toTuple() if node.color is not None else None
            raw.append((node.name, shape, col))
    else:
        raw.append(("body", obj, None))
    if not raw:
        raise ValueError("build() returned no parts")
    if len(raw) > 40:
        raise ValueError("build() returned more than 40 parts")
    used = set()
    out = []
    for i, (name, shape, col) in enumerate(raw):
        clean = safe_part_name(name, used)
        out.append((clean, _solid_of(shape, clean), normalize_color(col, i, len(raw))))
    return out

def _result_out(obj):
    parts = _parts_of(obj)
    pdir = os.path.join(build_dir, PARTS_DIR)
    os.makedirs(pdir, exist_ok=True)
    listed = []
    for name, val, color in parts:
        pstep = os.path.join(pdir, name + ".step")
        pstl = os.path.join(pdir, name + ".stl")
        cq.exporters.export(cq.Workplane("XY").add(val), pstep)
        cq.exporters.export(cq.Workplane("XY").add(val), pstl)
        listed.append({"name": name, "color": color, "stl_path": pstl, "step_path": pstep})
    # The whole model as one STEP that keeps part names and colours.
    step = os.path.join(build_dir, "result.step")
    if len(parts) == 1:
        cq.exporters.export(cq.Workplane("XY").add(parts[0][1]), step)
        whole = parts[0][1]
    else:
        assy = cq.Assembly(name="model")
        for name, val, color in parts:
            h = color.lstrip("#")
            assy.add(val, name=name, color=cq.Color(int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, 1))
        assy.export(step)
        whole = cq.Compound.makeCompound([v for _, v, _ in parts])
    bundle = write_bundle(listed, pdir)
    # Keep the long-standing result.* names the download and verification code read.
    import shutil
    stl = os.path.join(build_dir, "result.stl"); shutil.move(bundle["stl_path"], stl)
    glb = os.path.join(build_dir, "result.glb"); shutil.move(bundle["glb_path"], glb)
    tmf = os.path.join(build_dir, "result.3mf"); shutil.move(bundle["threemf_path"], tmf)
    bb = whole.BoundingBox()
    vol = sum(float(v.Volume()) for _, v, _ in parts)
    return {"ok": True, "step_path": step, "stl_path": stl, "glb_path": glb, "threemf_path": tmf,
            "bbox": [float(bb.xlen), float(bb.ylen), float(bb.zlen)], "volume": vol,
            "parts": bundle["parts"]}
'''
    tail = '''
_g = {"__builtins__": _allowed_builtins}
try:
    exec(compile(script, "<llm_build>", "exec"), _g, _g)
    if "build" not in _g:
        raise ValueError("script did not define build()")
    _res = _g["build"](params)
    _out = _result_out(_res)
    print(_cfg["marker"] + json.dumps(_out))
except Exception as _e:
    print(_cfg["marker"] + json.dumps({"ok": False, "error": type(_e).__name__ + ": " + str(_e)}))
'''
    return common + body + guard + tail


_PASSTHROUGH_ENV = (
    "PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP",
    "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA",
    "HOME", "LANG", "LC_ALL",
)


def _sandbox_ids() -> tuple[int, int] | None:
    """The unprivileged uid/gid builds run as."""
    if os.name != "posix" or os.geteuid() != 0:
        return None
    from textocad.core.config import settings
    uid = int(settings.sandbox_uid or 0)
    return (uid, uid) if uid > 0 else None


def _hand_over(path: Path, ids: tuple[int, int]) -> None:
    os.chown(path, *ids)
    for item in path.rglob("*"):
        try:
            os.chown(item, *ids, follow_symlinks=False)
        except OSError:
            pass


def run_local(mode: str, cfg: dict, timeout_s: int) -> tuple[str, str]:
    """Run the harness for one validated script in a child process and return its (stdout, stderr)."""
    bdir = Path(cfg["build_dir"])
    env = {k: os.environ[k] for k in _PASSTHROUGH_ENV if k in os.environ}
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("MALLOC_ARENA_MAX", "2")

    preexec = None
    ids = _sandbox_ids()
    if ids:
        _hand_over(bdir, ids)
        env["HOME"] = str(bdir)

        def preexec() -> None:
            os.setgroups([])
            os.setgid(ids[1])
            os.setuid(ids[0])

    proc = subprocess.run(
        [sys.executable, "-c", _runner_source(mode)],
        input=json.dumps(cfg), capture_output=True, text=True,
        timeout=timeout_s, cwd=str(bdir), env=env,
        preexec_fn=preexec,  # noqa: PLW1509 - drops privileges only, no shell
    )
    return proc.stdout or "", proc.stderr or ""


def _inside(path: str | None, root: Path) -> str | None:
    if not path:
        return None
    resolved = Path(path).resolve()
    return str(resolved) if root in resolved.parents else None


def execute_script(
    source: str,
    params: dict,
    mode: str = "precision",
    *,
    build_dir: str | None = None,
    timeout_s: int = 60,
    mem_mb: int | None = None,
) -> ExecResult:
    """Validate, then run the script in a hardened subprocess."""
    from textocad.core.config import settings

    if mem_mb is None:
        mem_mb = settings.sandbox_mem_mb
    v = validate_script(source, mode)
    if not v.ok:
        return ExecResult(ok=False, error="AST validation failed: " + "; ".join(v.errors))

    bdir = build_dir or tempfile.mkdtemp(prefix="codegen_")
    Path(bdir).mkdir(parents=True, exist_ok=True)
    bdir = str(Path(bdir).resolve())

    marker = "__RESULT_" + uuid.uuid4().hex + "__"
    cfg = {
        "script": source, "params": params, "build_dir": bdir,
        "mem_mb": mem_mb, "cpu_s": timeout_s,
        "harness_root": str(Path(__file__).resolve().parents[2]),
        "marker": marker,
        "allowed_roots": sorted(allowed_imports(mode)),
        "blocked_attrs": sorted(a for a in _BANNED_ATTRS if not a.startswith("__")),
    }

    try:
        stdout, stderr = run_local(mode, cfg, timeout_s)
    except subprocess.TimeoutExpired:
        return ExecResult(ok=False, error=f"execution timed out after {timeout_s}s (killed)")

    line = next((l for l in stdout.splitlines() if l.startswith(marker)), None)
    if line is None:
        tail = (stderr or stdout)[-500:]
        return ExecResult(ok=False, error=f"no result from sandbox; stderr: {tail}", stdout=stdout)

    data = json.loads(line[len(marker):])
    if not data.get("ok"):
        return ExecResult(ok=False, error=str(data.get("error", "unknown build error")), stdout=stdout)

    root = Path(bdir)
    for key in ("step_path", "stl_path", "glb_path", "threemf_path"):
        data[key] = _inside(data.get(key), root)
    for part in data.get("parts") or []:
        if isinstance(part, dict):
            for key in ("stl_path", "step_path", "glb_path"):
                if key in part:
                    part[key] = _inside(part.get(key), root)

    bbox = data.get("bbox")
    return ExecResult(
        ok=True,
        step_path=data.get("step_path"),
        stl_path=data.get("stl_path"),
        glb_path=data.get("glb_path"),
        bbox=tuple(bbox) if bbox else None,
        volume=data.get("volume"),
        stdout=stdout,
        parts=list(data.get("parts") or []),
        threemf_path=data.get("threemf_path"),
    )
