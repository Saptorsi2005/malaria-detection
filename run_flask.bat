@echo off
echo ============================================
echo   Malaria Detection - Flask Web Server
echo ============================================
echo.

REM Use the short-path venv at C:\mlenv (has TF 2.13, avoids Windows 260-char path limit)
set PYTHON=C:\mlenv\Scripts\python.exe

REM Fix protobuf compatibility issue with TF 2.10
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

REM Force CPU-only (prevents GPU detection hang)
set CUDA_VISIBLE_DEVICES=-1
set TF_CPP_MIN_LOG_LEVEL=3
set TF_XLA_FLAGS=--tf_xla_auto_jit=0

REM Check Python exists
if not exist "%PYTHON%" (
    echo ERROR: Python not found at %PYTHON%
    pause
    exit /b 1
)

echo Using Python: %PYTHON%
echo Open browser at: http://127.0.0.1:5000
echo First prediction takes 1-2 min (model loading) - please wait
echo.

cd /d "%~dp0"
"%PYTHON%" -m flask --app app run --no-reload

pause
