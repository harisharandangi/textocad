# Contributing to TextoCAD

Thanks for helping. Bug reports, failing prompts, docs fixes and pull requests are all welcome.

## Ways to help

- **Report a bad model.** Open an issue using the *Bad model* template. Include the prompt, the mode, the provider and model, and what was wrong (a missing hole, a wrong size, a build error). Attach the `result.json` and script if you can.
- **Report a bug.** Use the *Bug report* template, with the command you ran and the full error.
- **Improve the engine.** Prompt templates, repair checks, exporters, provider adapters, the local web UI.
- **Add tests.** Prompts with measurable expectations (for example, "4 holes, 60 mm wide") make every change safer.
- **Share examples.** A model you're proud of, with its prompt, makes the README better.

Not sure where to start? Open an issue and ask.

## Development setup

The quickest way is the start script. It creates `.venv` with an editable install, so your code changes apply on the next start:

```bash
git clone https://github.com/harisharandangi/textocad.git
cd textocad
./start.sh                         # Windows: start.bat
```

Or set it up by hand (Python 3.11 or 3.12, in the top folder that contains `pyproject.toml`):

```bash
python3.12 -m venv .venv           # Windows: py -3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[server,gears]"
```

Add your API key in the web page, or put it in `.env` (copy `.env.example`) for command-line use.

For `printing` mode, install a recent [OpenSCAD](https://openscad.org/downloads.html) snapshot and run `textocad fetch-libs`.

Check that everything works:

```bash
textocad generate "20 mm calibration cube with a 5 mm hole" --mode precision --out tmp/cube
textocad serve
```

## Project layout

| Path | What lives there |
|---|---|
| `textocad/cli.py` | The `textocad` command |
| `textocad/generate.py` | Public `generate()` and `rebuild()` entry points |
| `textocad/server.py`, `textocad/web/` | Local HTTP API and web page |
| `textocad/services/codegen_pipeline.py` | Prompt to script to build, per mode |
| `textocad/services/codegen_sandbox.py` | Script checks and the isolated build process |
| `textocad/services/openscad_engine.py` | OpenSCAD builds (printing mode) |
| `textocad/services/robot_pipeline.py` | Robotics mode and URDF |
| `textocad/services/verification.py`, `repair_loop.py` | Measure against the prompt and repair |
| `textocad/services/providers.py`, `llm_service.py` | LLM providers |

## Pull requests

1. Fork the repo and create a branch from `main`.
2. Keep each pull request focused on one change.
3. Test your change with a real prompt in the mode it affects. Describe what you ran in the PR.
4. Follow the existing style: clear names, small functions, and comments only where the code can't explain itself.
5. Don't commit API keys, `.env` files or generated models (`builds/`, `*.step`, `*.stl`, `*.glb`).

### Changes to the sandbox

`codegen_sandbox.py` decides what AI-written code is allowed to do. Pull requests that widen the import allowlist, remove banned names or relax resource limits need a clear reason and will get a careful review. If you find a way around the checks, please **don't** open a public issue or PR. Email security@textocad.com instead (see [SECURITY.md](SECURITY.md)).

## License of contributions

TextoCAD is distributed under the [PolyForm Noncommercial License 1.0.0](LICENSE), and GadiDokan Pvt. Ltd. also offers it under commercial licenses. By submitting a contribution, you confirm that you wrote it or have the right to submit it. You also agree that it may be distributed under the project's license and under those commercial licenses.

## Code of conduct

Be kind and constructive. Harassment or personal attacks aren't tolerated. Maintainers may remove comments, issues or contributors that break this.
