"""OpenSCAD libraries that generated scripts may include (BOSL2 only)."""

from __future__ import annotations

import hashlib
import io
import os
import threading
import urllib.request
import zipfile
from pathlib import Path

BOSL2_VERSION = "v2.0.779"
BOSL2_URL = f"https://github.com/BelfrySCAD/BOSL2/archive/refs/tags/{BOSL2_VERSION}.zip"
# Pinned so a changed or tampered archive is refused instead of installed.
BOSL2_SHA256 = "bfea288fe0db97d0268ad6cd9d2aead2c71b2114d8e97fc2950ca79f26c70709"

_lock = threading.Lock()


def lib_root() -> Path:
    """Where libraries live: TEXTOCAD_LIB_DIR, else ~/.textocad/openscad."""
    custom = os.getenv("TEXTOCAD_LIB_DIR", "").strip()
    return Path(custom).expanduser() if custom else Path.home() / ".textocad" / "openscad"


def bosl2_installed() -> bool:
    stamp = lib_root() / "BOSL2" / ".version"
    return stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == BOSL2_VERSION


def fetch_bosl2(force: bool = False) -> Path:
    """Download BOSL2 into lib_root() if it is missing. Returns its folder."""
    target = lib_root() / "BOSL2"
    with _lock:
        if bosl2_installed() and not force:
            return target
        data = urllib.request.urlopen(BOSL2_URL, timeout=120).read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != BOSL2_SHA256:
            raise RuntimeError(f"BOSL2 download does not match the pinned checksum ({digest})")
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                # Top-level .scad files and the licence only, so no path can
                # point outside the target folder.
                _, _, name = info.filename.partition("/")
                if not name or "/" in name or "\\" in name:
                    continue
                if name.endswith(".scad") or name == "LICENSE":
                    (target / name).write_bytes(zf.read(info))
        (target / ".version").write_text(BOSL2_VERSION, encoding="utf-8")
    return target
