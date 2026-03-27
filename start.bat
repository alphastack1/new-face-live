@echo off
cd /d "%~dp0"

echo.
echo   ================================
echo     NewFace - Cosmetic Preview
echo     http://localhost:5959
echo     Press Ctrl+C to stop
echo   ================================
echo.

if exist venv\Scripts\python.exe (
    venv\Scripts\python.exe app.py
) else (
    python app.py
)
pause
