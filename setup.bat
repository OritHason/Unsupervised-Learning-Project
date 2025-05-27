@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing required packages...
pip install -r requirements.txt

echo Running the main script...
python main.py

echo.
echo ✅ Setup complete.
echo To activate the environment later, run:
echo    venv\Scripts\activate
echo.
pause
