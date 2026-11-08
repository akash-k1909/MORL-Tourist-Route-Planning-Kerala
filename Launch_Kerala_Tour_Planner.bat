@echo off
echo ========================================================
echo   Launching Multi-Objective RL Tourist Route Planner...
echo   Kerala 56-Node Real-World Tourism Benchmark
echo ========================================================

set APP_PATH=%~dp0morl_tourist_app.html

:: Try launching in Google Chrome if installed
if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" (
    start "" "%ProgramFiles%\Google\Chrome\Application\chrome.exe" "%APP_PATH%"
    exit /b
)
if exist "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" (
    start "" "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" "%APP_PATH%"
    exit /b
)
if exist "%LocalAppData%\Google\Chrome\Application\chrome.exe" (
    start "" "%LocalAppData%\Google\Chrome\Application\chrome.exe" "%APP_PATH%"
    exit /b
)

:: Fallback to default browser
start "" "%APP_PATH%"
