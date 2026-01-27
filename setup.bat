@echo off
echo ========================================
echo Sign Language Captioner - Setup Script
echo ========================================
echo.

echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Installing PyTorch CPU version for Windows...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To activate the virtual environment in the future, run:
echo   venv\Scripts\activate
echo.
echo To collect training data, run:
echo   python src\collect_data.py
echo.
echo To train the model, run:
echo   python src\train_demo.py
echo.
echo To launch the app, run:
echo   streamlit run app\streamlit_app.py
echo.
pause
