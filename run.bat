@echo off
setlocal EnableExtensions EnableDelayedExpansion
title AETHER-SLAM
cd /d "%~dp0"

rem ======================================================================
rem  AETHER-SLAM launcher
rem
rem    run.bat                      start with the webcam (sets everything up on first run)
rem    run.bat --replay <folder>    replay a TUM RGB-D / Replica folder or a video file
rem    run.bat --cap4               emulate a 4 GB GPU (caps PyTorch memory)
rem    run.bat --port 8080          use another port (default 8000)
rem    run.bat --setup              reinstall the Python packages
rem    run.bat --rebuild            rebuild the web interface
rem    run.bat --no-browser         do not open the browser
rem    run.bat --help
rem
rem  Options can be combined:  run.bat --replay D:\tum\rgbd_dataset_freiburg1_desk --cap4
rem ======================================================================

set "PORT=8000"
set "FORCE_SETUP=0"
set "FORCE_BUILD=0"
set "OPEN_BROWSER=1"
set "REPLAY="
set "CAP="

:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--help"       goto usage
if /i "%~1"=="-h"           goto usage
if /i "%~1"=="--setup"      (set "FORCE_SETUP=1" & shift & goto parse)
if /i "%~1"=="--rebuild"    (set "FORCE_BUILD=1" & shift & goto parse)
if /i "%~1"=="--no-browser" (set "OPEN_BROWSER=0" & shift & goto parse)
if /i "%~1"=="--cap4"       (set "CAP=4" & shift & goto parse)
if /i "%~1"=="--port" (
    if "%~2"=="" (echo [ERROR] --port needs a number. & goto fail)
    set "PORT=%~2" & shift & shift & goto parse
)
if /i "%~1"=="--replay" (
    if "%~2"=="" (echo [ERROR] --replay needs a folder or a video file. & goto fail)
    if not exist "%~2" (echo [ERROR] Not found: %~2 & goto fail)
    set "REPLAY=%~f2" & shift & shift & goto parse
)
echo [ERROR] Unknown option: %~1
goto usage

:parsed
echo ======================================================================
echo                 AETHER-SLAM  //  dense monocular SLAM
echo ======================================================================

rem ---------------------------------------------------------------- 1. Python environment
set "PY=venv\Scripts\python.exe"
if exist "%PY%" (
    "%PY%" -c "import sys" >nul 2>&1
    if errorlevel 1 (
        echo [WARN] The venv's base Python cannot start ^(moved, removed, or blocked by Smart App Control^).
        call :repair_venv || goto fail
    )
)
if not exist "%PY%" (
    echo [1/4] Creating the Python environment...
    call :find_python || goto fail
    echo       using !BASEPY!
    "!BASEPY!" -m venv venv || goto fail
    set "FORCE_SETUP=1"
) else (
    echo [1/4] Python environment found.
)

rem reinstall the packages only when requirements.txt changed or --setup was given
if "%FORCE_SETUP%"=="0" if not exist "venv\.requirements.installed" set "FORCE_SETUP=1"
if "%FORCE_SETUP%"=="0" (
    fc /b requirements.txt "venv\.requirements.installed" >nul 2>&1 || set "FORCE_SETUP=1"
)
if "%FORCE_SETUP%"=="1" (
    echo [2/4] Installing Python packages ^(the first run takes several minutes^)...
    "%PY%" -m pip install --upgrade pip >nul
    "%PY%" -m pip install -r requirements.txt || (echo [ERROR] pip install failed. & goto fail)
    copy /y requirements.txt "venv\.requirements.installed" >nul
) else (
    echo [2/4] Python packages up to date.
)

rem PyTorch: CUDA 12.1 build when an NVIDIA GPU is present (versions pinned as in the Dockerfile)
set "HAS_GPU=0"
nvidia-smi >nul 2>&1 && set "HAS_GPU=1"
"%PY%" -c "import torch, sys; sys.exit(0 if (torch.cuda.is_available() or %HAS_GPU%==0) else 1)" >nul 2>&1
if errorlevel 1 (
    if "%HAS_GPU%"=="1" (
        echo       Installing PyTorch 2.5.1 with CUDA 12.1 ^(about 2.5 GB^)...
        "%PY%" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall || goto fail
    ) else (
        echo       No NVIDIA GPU found: installing CPU PyTorch ^(this will be slow^)...
        "%PY%" -m pip install torch==2.5.1 torchvision==0.20.1 || goto fail
    )
)
"%PY%" -c "import torch; print('      PyTorch', torch.__version__, '- GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU)')" 2>nul

rem ---------------------------------------------------------------- 2. Web interface
set "NEED_BUILD=%FORCE_BUILD%"
if not exist "backend\static\index.html" set "NEED_BUILD=1"
if "%NEED_BUILD%"=="0" (
    rem rebuild when a frontend source file is newer than the built page
    "%PY%" -c "import os,sys;s=[os.path.join(r,f) for r,_,fs in os.walk('frontend/src') for f in fs]+['frontend/index.html','frontend/package.json'];sys.exit(1 if max(os.path.getmtime(p) for p in s if os.path.exists(p))>os.path.getmtime('backend/static/index.html') else 0)" || set "NEED_BUILD=1"
)
if "%NEED_BUILD%"=="0" (
    echo [3/4] Web interface up to date.
    goto start_server
)
node --version >nul 2>&1
if errorlevel 1 (
    if exist "backend\static\index.html" (
        echo [3/4] Node.js not found: using the existing web interface.
        goto start_server
    )
    echo [ERROR] Node.js is needed once to build the web interface: https://nodejs.org
    goto fail
)
echo [3/4] Building the web interface...
pushd frontend
if not exist node_modules (
    call npm install
    if errorlevel 1 (popd & goto fail)
)
call npm run build
if errorlevel 1 (popd & goto fail)
popd
if exist backend\static rd /s /q backend\static
xcopy /e /i /y /q frontend\dist backend\static >nul || goto fail

rem ---------------------------------------------------------------- 3. Start
:start_server
netstat -ano | findstr /r /c:":%PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [ERROR] Port %PORT% is already in use ^(is AETHER-SLAM already running?^). Try: run.bat --port 8080
    goto fail
)
set "AETHER_PORT=%PORT%"
set "PYTHONIOENCODING=utf-8"
if defined REPLAY (set "AETHER_SOURCE=%REPLAY%") else (set "AETHER_SOURCE=webcam")
if not defined REPLAY echo       No webcam? Replay a dataset instead:  run.bat --replay ^<TUM or Replica folder^>
if defined CAP (set "AETHER_GPU_CAP_GB=%CAP%") else (set "AETHER_GPU_CAP_GB=")

echo [4/4] Starting the server ^(loading the networks takes a few seconds^)
echo       camera : %AETHER_SOURCE%
if defined CAP echo       GPU memory capped at %CAP% GB
echo       open   : http://localhost:%PORT%      Ctrl+C stops the server
echo.
if "%OPEN_BROWSER%"=="1" (
    rem open the browser as soon as the server answers
    start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 300;$i++){try{Invoke-WebRequest -UseBasicParsing http://localhost:%PORT%/api/status -TimeoutSec 2 | Out-Null; Start-Process 'http://localhost:%PORT%'; break}catch{Start-Sleep 1}}"
)
"%PY%" -m backend.main
set "RC=%errorlevel%"
echo.
echo Server stopped ^(exit code %RC%^).
pause
endlocal & exit /b %RC%

rem ======================================================================  helpers
:find_python
rem Sets BASEPY to a runnable Python. Prefers the signed python.org builds, because
rem Smart App Control blocks unsigned interpreters (e.g. uv-managed Python), then the
rem py launcher, then python on PATH. PyTorch 2.5.1 needs Python 3.10-3.12.
rem Optional argument: one version only (e.g. 3.10).
set "BASEPY="
if "%~1"=="" (set "CANDS=3.10 3.11 3.12") else (set "CANDS=%~1")
for %%v in (%CANDS%) do if not defined BASEPY (
    set "VV=%%v"
    set "VV=!VV:.=!"
    for %%p in ("%LOCALAPPDATA%\Programs\Python\Python!VV!\python.exe" "%ProgramFiles%\Python!VV!\python.exe") do (
        if not defined BASEPY if exist "%%~p" (
            "%%~p" -c "import sys" >nul 2>&1 && set "BASEPY=%%~p"
        )
    )
    if not defined BASEPY (
        for /f "delims=" %%x in ('py -%%v -c "import sys; print(sys.executable)" 2^>nul') do set "BASEPY=%%x"
    )
)
if not defined BASEPY if "%~1"=="" (
    python -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>&1
    if not errorlevel 1 for /f "delims=" %%x in ('python -c "import sys; print(sys.executable)"') do set "BASEPY=%%x"
)
if defined BASEPY exit /b 0
echo [ERROR] Python %CANDS% not found ^(PyTorch 2.5.1 does not support newer versions^).
echo         Install it with:  winget install --id Python.Python.3.10 --scope user
exit /b 1

:repair_venv
rem The installed packages are built for Python 3.10: point the venv at a working Python 3.10.
call :find_python 3.10 || (
    echo         Or delete the venv folder and run this script again to rebuild it.
    exit /b 1
)
for %%d in ("!BASEPY!") do set "BASEDIR=%%~dpd"
set "BASEDIR=!BASEDIR:~0,-1!"
for /f "tokens=2" %%v in ('""!BASEPY!" --version"') do set "BASEVER=%%v"
copy /y venv\pyvenv.cfg venv\pyvenv.cfg.bak >nul 2>&1
(
    echo home = !BASEDIR!
    echo include-system-site-packages = false
    echo version = !BASEVER!
) > venv\pyvenv.cfg
"%PY%" -c "import sys" >nul 2>&1 || (echo [ERROR] Repair failed; the old file is venv\pyvenv.cfg.bak. & exit /b 1)
echo       venv repaired: it now uses !BASEPY!
exit /b 0

:usage
echo.
echo Usage: run.bat [options]
echo   --replay ^<folder^|video^>  replay a TUM RGB-D / Replica sequence or a video instead of the webcam
echo   --cap4                    emulate a 4 GB GPU
echo   --port ^<n^>                server port ^(default 8000^)
echo   --setup                   reinstall the Python packages
echo   --rebuild                 rebuild the web interface
echo   --no-browser              do not open the browser automatically
echo.
echo Example: run.bat --replay C:\data\rgbd_dataset_freiburg1_desk --cap4
endlocal & exit /b 0

:fail
echo.
echo Startup failed. See the messages above.
pause
endlocal & exit /b 1
