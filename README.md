<div align="center">

<img src="assets/logo.png" alt="TextoCAD logo" width="140">

# TextoCAD

### Free, source-available text to CAD engine: describe a part in plain English, get an editable parametric model

Generates CadQuery and OpenSCAD scripts with named dimensions and exports **STEP, STL, GLB and 3MF** for 3D printing, mechanical design and robotics.

[![Try it online](https://img.shields.io/badge/Try_it_free-textocad.com-2563eb?style=for-the-badge)](https://textocad.com)

[![License: PolyForm Noncommercial](https://img.shields.io/badge/license-PolyForm%20NC%201.0.0-blue)](LICENSE)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-3776ab)](https://www.python.org/)
[![CadQuery](https://img.shields.io/badge/CAD-CadQuery%20%7C%20OpenSCAD-orange)](#how-it-works)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](CONTRIBUTING.md)

<img src="assets/demo/cup.gif" alt="TextoCAD turning a text prompt into a parametric mug with an elliptical handle" width="820">

**[Try it on textocad.com](https://textocad.com)** · [Examples](#examples) · [Quick start](#quick-start) · [Troubleshooting](#troubleshooting) · [Contribute](#contributing)

</div>

---

## Why TextoCAD

AI image and mesh generators give you a blob you can't edit. TextoCAD writes **real CAD code** instead, so you get a model you can measure, change and manufacture.

- 📐 **Parametric by default.** Every model is a `build(params)` script with named dimensions. Change `wall_thickness` from 2 to 3 mm and rebuild in seconds, with no new AI call.
- 🧩 **Four modes for different jobs.**

  | Mode | Kernel | Best for |
  |---|---|---|
  | `printing` | OpenSCAD + BOSL2 | FDM-friendly parts, brackets, enclosures |
  | `precision` | CadQuery / OpenCASCADE | Real B-rep solids and STEP files for CAD tools |
  | `organic` | SDF to mesh | Sculpted, smooth, freeform shapes |
  | `robotics` | Jointed assembly | Robot arms with movable joints and URDF export |

- ✅ **Checks its own work.** With `--details`, it pulls the measurable requirements out of your prompt (hole count, tooth count, overall size, fillets), measures the built part and regenerates if it doesn't match.
- 🔑 **Bring any model.** Works with OpenAI, Anthropic (Claude), Google Gemini, xAI, DeepSeek, OpenRouter, Mistral, Groq, Together and Fireworks keys.
- 🖥️ **Local web app included.** Chat to design and change a part, turn and inspect it in 3D, drag dimension sliders, move robot joints and download files, all on your own machine.

> **Just want to make parts?** The hosted version at **[textocad.com](https://textocad.com)** needs no install. It adds a full browser editor, a feature tree, a model gallery, and connectors for Claude and ChatGPT. You can try it free.

## Examples

All of these were generated from the prompt shown, on [textocad.com](https://textocad.com). Click a preview to watch the full video.

<table>
<tr>
<td width="50%" valign="top">

<a href="assets/demo/cup.mp4"><img src="assets/demo/cup.gif" alt="AI-generated parametric mug with an elliptical handle in CadQuery"></a>

**Mug with an elliptical handle**
> Create a cup with a handle on it. Cup should have a small base and a slightly larger, rounded top with an elliptical handle on it.

</td>
<td width="50%" valign="top">

<a href="assets/demo/bottle.mp4"><img src="assets/demo/bottle.gif" alt="AI-generated bottle with a threaded cap, editable with dimension sliders"></a>

**Bottle with a cap**
> Create a bottle.

Even a three-word prompt becomes a two-part model with sliders for body diameter, height, shoulder and neck.

</td>
</tr>
<tr>
<td width="50%" valign="top">

<a href="assets/demo/mouse.mp4"><img src="assets/demo/mouse.gif" alt="AI-generated wireless mouse enclosure split into top and bottom shells for 3D printing"></a>

**Wireless mouse enclosure, print-ready**
> Create a wireless computer mouse enclosure with a smooth, ergonomic, symmetrical shape, 115 × 65 × 38 mm, 2 mm walls…

<details>
<summary>Full prompt</summary>

> Create a wireless computer mouse enclosure with a smooth, ergonomic, symmetrical shape. Dimensions: Length 115 mm, Width 65 mm, Maximum height 38 mm, Wall thickness 2 mm, rounded outer edges and corners. Include: left and right mouse buttons separated by a 1.5 mm gap; central scroll-wheel opening 12 mm wide and 28 mm long; a small DPI button opening behind the scroll wheel; flat bottom surface; optical sensor opening on the bottom; USB receiver storage compartment on the bottom; battery compartment for one AA battery; removable bottom battery cover; four shallow recesses for mouse-glide pads. Do not add a cable or cable opening. The model should represent a wireless mouse. Create the enclosure as separate top and bottom parts and arrange them next to each other for 3D printing.

</details>

</td>
<td width="50%" valign="top">

<a href="assets/demo/robotic-arm.mp4"><img src="assets/demo/robotic-arm.gif" alt="AI-generated six-axis robotic arm with movable joints and URDF export"></a>

**Robotic arm with movable joints**
> Create a robotic arm.

Robotics mode builds a 7-link arm with joint limits you can drag, and exports URDF.

</td>
</tr>
</table>

<table>
<tr>
<td width="33%" valign="top"><img src="assets/demo/helical-spring.png" alt="AI-generated hollow helical compression spring on adjustable mounting legs"><br><b>Spring on adjustable legs</b> (37 parts)<br><sub>“Create a large hollow helical compression spring with five thick rounded coils, supported above the ground by three evenly spaced adjustable metal mounting legs with square base plates and triangular braces.”</sub></td>
<td width="33%" valign="top"><img src="assets/demo/core-shell-sphere.png" alt="AI-generated core-shell particle cross-section, sphere inside a hollow sphere"><br><b>Core-shell particle cross-section</b><br><sub>“Model a simple sphere inside a larger hollow sphere to visually represent the core urea-zeolite matrix and the biodegradable sodium alginate outer coating.”</sub></td>
<td width="33%" valign="top"><img src="assets/demo/steel-pergola.png" alt="AI-generated 9 by 12 metre black steel pergola CAD model"><br><b>9 × 12 m steel pergola</b><br><sub>“Create a large rectangular black steel pergola measuring 9000 × 12000 × 2777 mm, with six vertical support posts, a reinforced perimeter frame, evenly spaced parallel roof slats, and three long central support beams.”</sub></td>
</tr>
</table>

## Quick start

**You need:** [Python 3.11 or 3.12](https://www.python.org/downloads/) (3.13+ is not supported yet), [Git](https://git-scm.com/downloads), and an API key from any [supported AI provider](#configuration).

**1. Download**

```bash
git clone https://github.com/harisharandangi/textocad.git
```

No Git? Click **Code → Download ZIP** on this page and unzip it.

**2. Start**

- **Windows:** open the downloaded folder and double-click **`start.bat`**
- **macOS / Linux:** open a terminal in the downloaded folder and run `./start.sh`

The first run sets everything up (a few minutes). After that it starts in seconds. Your browser opens at **http://127.0.0.1:8000**.

**3. Paste your API key in the page and describe a part.** The key is checked with your provider and saved only on your computer, in `~/.textocad/credentials.json`. Change or remove it any time with the key button at the top.

That's it. To stop TextoCAD, close the window (or press `Ctrl+C`). To start again, run the same file.

> Port 8000 busy? Run `start.bat --port 8080` or `./start.sh --port 8080`.

### Optional: OpenSCAD for `printing` mode

`precision`, `organic` and `robotics` work out of the box. `printing` mode also needs OpenSCAD. Install a **development snapshot** from [openscad.org](https://openscad.org/downloads.html#snapshots), which is much faster than the 2021 release.

| OS | After installing |
|---|---|
| Windows | Nothing. It's found in `C:\Program Files\OpenSCAD (Nightly)` or `C:\Program Files\OpenSCAD` |
| macOS | Set `OPENSCAD_BIN=/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD` |
| Linux | Make sure `openscad` is on your PATH, or set `OPENSCAD_BIN` |

The [BOSL2](https://github.com/BelfrySCAD/BOSL2) library downloads by itself the first time it's needed (pinned and checksum-verified).

### Other ways to install

<details>
<summary><b>As a command with uv</b> (no clone needed)</summary>

[uv](https://docs.astral.sh/uv/getting-started/installation/) also downloads the right Python for you.

```bash
uv tool install --python 3.12 "textocad[server] @ git+https://github.com/harisharandangi/textocad.git"
textocad serve
```

Update with `uv tool upgrade textocad`, remove with `uv tool uninstall textocad`.

</details>

<details>
<summary><b>Manually, for development</b></summary>

Run these in the repo's top folder (the one with `pyproject.toml`).

```bash
python3.12 -m venv .venv                  # Windows: py -3.12 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[server,gears]"
textocad serve
```

The `gears` extra adds [cq_gears](https://github.com/meadiode/cq_gears) for accurate involute gears.

</details>

### Update

```bash
git pull
```

Then start as usual. The start script reinstalls only when dependencies change.

## Usage

### Command line

If you used the start script, the command is inside the `.venv` folder: `.venv\Scripts\textocad` on Windows, `.venv/bin/textocad` on macOS and Linux.

```bash
# generate a model
textocad generate "parametric phone stand, 75 degree angle" --mode printing --out stand

# change dimensions and rebuild, no AI call
textocad rebuild stand/result.json --set angle=65 width=90

# plan, verify against the prompt and repair (slower, more accurate)
textocad generate "spur gear, module 2, 24 teeth, 8 mm bore" --mode precision --details
```

Each run writes the script, `result.json` (parameters and metadata) and the exported files to the output folder.

### Local web app and HTTP API

`textocad serve` starts a page at http://127.0.0.1:8000 with three panels:

- **Chat (right):** describe a part, then keep typing to change it ("make it 10 mm taller"). Every version stays in the chat so you can go back to it.
- **Viewer (middle):** orbit, pan, zoom to the cursor, fit, standard views, orthographic mode, edges, X-ray, grid and a saved picture. Click a part to pick it. **Parts** shows, hides and recolours parts; **Look** changes lighting, material and shadows. Robots get a **Move joints** panel with a motion preview.
- **Model (left):** size and volume, dimension sliders that rebuild without an AI call, and downloads (STEP, STL, 3MF, GLB, plus URDF and a ROS 2 package for robots).

Shortcuts: `F` fit, `0` iso, `1` front, `3` right, `7` top, `O` ortho, `E` edges, `X` X-ray, `G` grid, `Space` play robot motion.

The same server answers `POST /generate` (pass `parentId` to change an earlier build) and `POST /builds/{id}/rebuild`.

### Python

```python
from textocad.generate import generate

result = generate("M8 flange bearing housing, 60 mm square base, four 6.5 mm holes",
                  mode="precision", out_dir="housing")
print(result.files["step"], result.params)
```

## How it works

```
prompt ──► mode router ──► LLM writes a build(params) script ──► sandboxed build ──► STEP / STL / GLB / 3MF
                                    ▲                                    │
                                    └──── measure vs. requirements ◄─────┘   (--details)
```

1. **Route.** The prompt is matched to a mode, or you choose one with `--mode`.
2. **Write.** The model writes a CadQuery or OpenSCAD script with a parameter list (name, value, min, max, unit).
3. **Build.** The script runs in a separate process with an import allowlist and memory and CPU limits.
4. **Verify (optional).** Measurable requirements are checked against the real geometry, and failures are fed back for a repair pass.
5. **Export.** You get STEP for CAD tools, STL and 3MF for slicers, and GLB for the web.

## Configuration

Most people only need to paste a key in the web page. A key saved there takes priority over everything below.

Supported providers: **OpenAI, Anthropic (Claude), Google Gemini, xAI, DeepSeek, OpenRouter, Mistral, Groq, Together and Fireworks.**

For scripts and servers, settings can also come from environment variables or a `.env` file in the folder you run from (see [.env.example](.env.example)):

| Variable | Purpose |
|---|---|
| `TEXTOCAD_API_KEY` | Your key for any supported provider |
| `TEXTOCAD_PROVIDER` | Optional. Detected from the key when empty: `openai`, `anthropic`, `gemini`, `xai`, `deepseek`, `openrouter`, `mistral`, `groq`, `together`, `fireworks` |
| `TEXTOCAD_MODEL` | Optional. Defaults to the provider's strongest chat model |
| `DEEPSEEK_API_KEY` | Use the built-in DeepSeek engine instead |
| `GENERATION_BUDGET_SECONDS` | Hard time limit for one generation (default 420) |
| `TEXTOCAD_LIB_DIR` | Where BOSL2 is stored |
| `OPENSCAD_BIN` | Full path to OpenSCAD if it is not found automatically |
| `TEXTOCAD_HOME` | Where the key saved from the web page is kept (default `~/.textocad`) |

## Troubleshooting

| You see | Fix |
|---|---|
| `TextoCAD needs Python 3.11 or 3.12` | Install [Python 3.12](https://www.python.org/downloads/) (tick **Add to PATH** on Windows), or install [uv](https://docs.astral.sh/uv/), then run the start file again |
| `start.bat` closes straight away | Open a terminal in the folder and run `start.bat` there to read the error |
| `permission denied: ./start.sh` | `chmod +x start.sh`, then run it again |
| `No module named venv` / `ensurepip is not available` (Linux) | `sudo apt install python3.12-venv` (or `python3-venv`) |
| `libGL.so.1: cannot open shared object file` (Linux) | `sudo apt install libgl1` |
| `git` is not recognized | Install [Git](https://git-scm.com/downloads), or use **Download ZIP** |
| The browser didn't open | Go to http://127.0.0.1:8000 yourself |
| Port 8000 is already in use | `start.bat --port 8080` or `./start.sh --port 8080` |
| The key isn't accepted | Choose the provider from the list instead of *Detect*, and check the key has credit or billing enabled |
| `OpenSCAD is not installed or not on PATH` | Install OpenSCAD (see above), or pick a mode other than *3D printing* |
| Printing mode is very slow | Use an OpenSCAD development snapshot instead of the 2021 release |
| Generation times out | Use a faster model, or raise `GENERATION_BUDGET_SECONDS` |
| Installing by hand: `does not appear to be a Python project` | You're in the inner `textocad/` folder. Run `cd ..` |
| Installing by hand: `requires a different Python` | Your Python is 3.13+. Make the venv with 3.12 (`py -3.12 -m venv .venv`) |

Still stuck? [Open an issue](https://github.com/harisharandangi/textocad/issues/new/choose) with the command and the full error.

## FAQ

**Do I need a GPU?** No. The AI runs at your provider, and the CAD build runs on your CPU.

**What does it cost?** The engine is free for noncommercial use. You pay your AI provider for the tokens you use. Or try [textocad.com](https://textocad.com) with no setup.

**Does it work offline?** No. It needs an AI provider's API. Rebuilding a model with new dimensions (`textocad rebuild`) works without one.

**Where are my files?** Use the download buttons in the page. They are also kept in `builds/<id>/` inside the TextoCAD folder, or wherever `--out` points.

**Where is my API key stored?** Only on your computer, in `~/.textocad/credentials.json`. It is sent only to the AI provider you chose. Remove it from the page with *Remove saved key*.

**Which AI model works best?** The provider's strongest model gives the best geometry. Smaller models are faster and cheaper but fail more often on complex parts.

**Can I use it at work?** Only for noncommercial purposes. Commercial use needs a license, see [License](#license).

## Contributing

Contributions are very welcome, whether that's a bug report, a prompt that produced a bad model, or a pull request. Good first areas:

- 🐛 **Prompts that fail.** Open an issue with the prompt, the mode and what went wrong. These help the most.
- 🧪 **Tests and benchmarks.** A set of prompts with measurable expectations helps everyone.
- 🔧 **New exporters, providers and repair checks.**
- 📝 **Docs and examples.** Share a model you made, with its prompt.

Read [CONTRIBUTING.md](CONTRIBUTING.md) to get set up. If you like the project, a ⭐ helps others find it.

## Security

The engine executes code written by a language model. Scripts are checked against an import allowlist and run with resource limits, but that is a seatbelt, not a sandbox. Keep `textocad serve` on `127.0.0.1`, and read [SECURITY.md](SECURITY.md) before exposing it to anyone else. Please report vulnerabilities privately to security@textocad.com.

## License

Source-available under the [PolyForm Noncommercial License 1.0.0](LICENSE).

- ✅ **Free** for personal projects, research, education, hobby use and other noncommercial purposes. You may use, study and modify it.
- 💼 **Commercial use needs a license.** That includes running it as a paid or ad-supported service, building it into a commercial product, or using it inside a for-profit company. Contact **licensing@textocad.com**.

"TextoCAD" and the TextoCAD logo are trademarks of GadiDokan Pvt. Ltd. and are not covered by the license. Third-party components are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

<div align="center">

**[textocad.com](https://textocad.com)**: AI text to CAD in your browser, free to try.

</div>
