@echo off
rem Sets up TextoCAD on first run, then starts it. Extra arguments go to "textocad serve".
setlocal
cd /d "%~dp0"
title TextoCAD

if not exist ".venv\Scripts\python.exe" goto setup
".venv\Scripts\python.exe" -c "import sys; sys.exit(sys.version_info[:2] not in ((3, 11), (3, 12)))" 2>nul
if not errorlevel 1 goto install
echo The .venv folder uses an unsupported Python version. Recreating it.
rmdir /s /q .venv

:setup
echo Setting up TextoCAD for the first time. This takes a few minutes.
where uv >nul 2>nul
if errorlevel 1 goto nouv
uv venv --seed --python 3.12 .venv
if errorlevel 1 goto fail
goto install

:nouv
py -3.12 -m venv .venv 2>nul && goto install
py -3.11 -m venv .venv 2>nul && goto install
python -c "import sys; sys.exit(sys.version_info[:2] not in ((3, 11), (3, 12)))" 2>nul && python -m venv .venv && goto install
echo.
echo TextoCAD needs Python 3.11 or 3.12.
echo Install Python 3.12 from https://www.python.org/downloads/ and run this again.
goto fail

:install
fc /b pyproject.toml .venv\installed.toml >nul 2>nul
if not errorlevel 1 goto run
set "EXTRAS=server"
where git >nul 2>nul
if not errorlevel 1 set "EXTRAS=server,gears"
echo Installing TextoCAD...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -e ".[%EXTRAS%]"
if errorlevel 1 goto fail
copy /y pyproject.toml .venv\installed.toml >nul

:run
echo Starting TextoCAD. Close this window to stop it.
".venv\Scripts\python.exe" -m textocad.cli serve %*
goto end

:fail
echo.
echo Setup did not finish. See the message above, or Troubleshooting in README.md.
pause
exit /b 1

:end
endlocal
