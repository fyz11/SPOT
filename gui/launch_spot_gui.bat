@echo off
rem Double-click to start the SPOT GUI.
rem Set SPOT_ENV below to the name of the conda environment where SPOT is installed.
set SPOT_ENV=SAM-SPOT_env

where conda >nul 2>nul
if %errorlevel%==0 (
    call conda activate %SPOT_ENV%
) else if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" (
    call "%USERPROFILE%\miniconda3\condabin\conda.bat" activate %SPOT_ENV%
) else if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
    call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate %SPOT_ENV%
)
python "%~dp0spot_gui.py" %*
if errorlevel 1 pause
