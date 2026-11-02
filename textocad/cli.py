from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from textocad.generate import MODES, generate, rebuild


def _params(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="textocad", description="Text to parametric CAD.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate a model from a prompt")
    gen.add_argument("prompt")
    gen.add_argument("--mode", choices=MODES)
    gen.add_argument("--out", default=None, help="output folder")
    gen.add_argument("--details", action="store_true", help="plan, verify and repair (slower)")
    gen.add_argument("--kernel", choices=("openscad", "cadquery"), default="openscad", help="robotics only")

    reb = sub.add_parser("rebuild", help="rebuild a generated model with new parameters")
    reb.add_argument("result", help="result.json from a previous generate")
    reb.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    reb.add_argument("--out", default=None)

    libs = sub.add_parser("fetch-libs", help="download the BOSL2 OpenSCAD library")
    libs.add_argument("--force", action="store_true", help="download again even if present")

    srv = sub.add_parser("serve", help="run the local HTTP API")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    srv.add_argument("--no-browser", action="store_true", help="don't open the web page")

    args = parser.parse_args(argv)

    if args.command == "fetch-libs":
        from textocad.libs import BOSL2_VERSION, fetch_bosl2

        print(f"BOSL2 {BOSL2_VERSION} -> {fetch_bosl2(force=args.force)}")
        return 0

    if args.command == "serve":
        import threading
        import webbrowser

        import uvicorn

        if not args.no_browser:
            shown = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
            threading.Timer(2.0, webbrowser.open, [f"http://{shown}:{args.port}"]).start()
        uvicorn.run("textocad.server:app", host=args.host, port=args.port)
        return 0

    if args.command == "generate":
        result = generate(args.prompt, args.mode, out_dir=args.out, extra_details=args.details, kernel=args.kernel)
    else:
        prev = json.loads(Path(args.result).read_text(encoding="utf-8"))
        result = rebuild(
            prev["script"], _params(args.set), prev["mode"],
            param_schema=prev.get("params"), out_dir=args.out,
        )

    if not result.ok:
        print(f"error: {result.error}", file=sys.stderr)
        return 1

    out = Path(next(iter(result.files.values()))).parent if result.files else Path(".")
    (out / "result.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"{result.title or 'model'} ({result.mode}) -> {out}")
    for kind, path in result.files.items():
        print(f"  {kind:7} {path}")
    for p in result.params:
        print(f"  param   {p.get('key')} = {p.get('value', p.get('defaultValue'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
