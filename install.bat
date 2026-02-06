@echo off
echo Installing AI Data Analysis Agent...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.10 or higher from https://www.python.org/
    pause
    exit /b 1
)

REM Check if virtual environment exists, if not create it
if not exist ".venv\" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment.
        echo Make sure you have venv module installed.
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo Error: Failed to activate virtual environment.
    pause
    exit /b 1
)

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo Warning: Some dependencies failed to install.
    echo Trying alternative installation method...
    
    echo Installing core dependencies individually...
    pip install agno streamlit duckdb pandas openpyxl pyarrow python-dotenv
    pip install "openai<1.0.0,>=0.28.0"
    
    if errorlevel 1 (
        echo Error: Failed to install dependencies.
        echo You may need to run as Administrator or use --user flag.
        echo.
        echo Try: pip install --user -r requirements.txt
        pause
        exit /b 1
    )
)

REM Test imports
echo Testing imports...
python test_imports.py
if errorlevel 1 (
    echo Warning: Some imports failed, but installation may still work.
    echo Check the error messages above.
)

echo.
echo Installation complete!
echo.
echo To run the basic application:
echo   streamlit run app.py
echo.
echo To run the enhanced application (recommended):
echo   streamlit run app_enhanced.py
echo.
echo Don't forget to set your DeepSeek API key in the .env file!
echo Get your API key from: https://platform.deepseek.com/
echo.
pause