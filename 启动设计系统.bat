@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHON_EXE="
set "PYTHON_ARGS="

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
    goto launch
)
if exist "%LOCALAPPDATA%\Python\bin\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Python\bin\python.exe"
    goto launch
)
where py >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_EXE=py"
    set "PYTHON_ARGS=-3"
    goto launch
)
where python >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_EXE=python"
    goto launch
)
echo 未找到 Python。请先安装带 Tkinter 的 Python 3.8 或更高版本。
pause
exit /b 1

:launch
"%PYTHON_EXE%" %PYTHON_ARGS% "%~dp0python\app_gui.py"
if errorlevel 1 (
    echo.
    echo 启动失败，请查看上方错误信息。
    echo 如缺少依赖，可在当前目录运行：
    echo "%PYTHON_EXE%" %PYTHON_ARGS% -m pip install -r requirements.txt
    pause
    exit /b 1
)
endlocal
exit /b 0
