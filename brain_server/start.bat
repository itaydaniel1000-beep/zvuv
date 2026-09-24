@echo off
rem Double-click to start the full-brain server for the zvuv site.
rem First run: creates a Python environment in .venv, installs packages and downloads
rem the FlyWire data (~140 MB). Later runs start straight away.
cd /d "%~dp0.."
if not exist .venv\installed.ok (
  echo Creating Python environment in .venv ...
  if not exist .venv\Scripts\python.exe (py -3 -m venv .venv || python -m venv .venv || goto nopython)
  .venv\Scripts\python.exe -m pip install --upgrade pip
  .venv\Scripts\python.exe -m pip install -r brain_server\requirements.txt || goto failed
  echo ok> .venv\installed.ok
)
.venv\Scripts\python.exe brain_server\server.py %*
pause
exit /b
:nopython
echo.
echo Python was not found. Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
pause
exit /b 1
:failed
echo.
echo Installing the packages failed. Check the internet connection and run start.bat again.
pause
exit /b 1
