@echo off

:: Check if the virtual environment folder already exists
IF NOT EXIST venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

:: Install the required dependencies
echo Installing dependencies...
pip install -r requirements.txt

:: Run the main Python script
echo Running the application...
python main.py

:: Deactivate the virtual environment after the script finishes
echo Deactivating virtual environment...
deactivate

pause