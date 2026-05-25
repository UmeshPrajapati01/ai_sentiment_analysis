@echo off
echo Installing Python dependencies...
echo.

cd /d "c:\ai_sentiment_analysis\ai_sentiment"

echo Installing from requirements.txt...
pip install -r requirements.txt

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ Installation completed successfully!
    echo.
) else (
    echo.
    echo ✗ Installation failed with errors
    echo.
)

pause
