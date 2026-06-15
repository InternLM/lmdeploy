@echo off
REM Test runner script with coverage for LMDeploy (Windows)
REM Usage: scripts\run_tests.bat [options]

setlocal enabledelayedexpansion

REM Default values
if not defined COVERAGE set COVERAGE=1
if not defined THRESHOLD set THRESHOLD=40
if not defined TEST_PATH set TEST_PATH=tests
if not defined PARALLEL set PARALLEL=1
if not defined NUM_WORKERS set NUM_WORKERS=4

echo ==========================================
echo LMDeploy Test Runner with Coverage
echo ==========================================
echo.

REM Build pytest command
set PYTEST_CMD=pytest

REM Add coverage options if enabled
if "%COVERAGE%"=="1" (
    echo Coverage reporting: ENABLED
    echo Coverage threshold: %THRESHOLD%%%
    set PYTEST_CMD=%PYTEST_CMD% --cov=lmdeploy --cov-report=term-missing --cov-report=html --cov-report=xml
) else (
    echo Coverage reporting: DISABLED
)

REM Add parallel execution if enabled
if "%PARALLEL%"=="1" (
    echo Parallel execution: ENABLED (%NUM_WORKERS% workers)
    set PYTEST_CMD=%PYTEST_CMD% -n %NUM_WORKERS%
) else (
    echo Parallel execution: DISABLED
)

REM Add test path and any additional arguments
set PYTEST_CMD=%PYTEST_CMD% %TEST_PATH% %*

echo.
echo Running: %PYTEST_CMD%
echo ==========================================
echo.

REM Run tests
call %PYTEST_CMD%

if errorlevel 1 (
    echo.
    echo ==========================================
    echo Tests failed!
    echo ==========================================
    exit /b 1
)

REM Check coverage threshold if coverage is enabled
if "%COVERAGE%"=="1" (
    echo.
    echo ==========================================
    echo Checking coverage threshold...

    python -c "import coverage; c = coverage.Coverage(); c.load(); total = c.report(); print(f'Total coverage: {total:.1f}%%'); import sys; sys.exit(0 if total >= %THRESHOLD% else 1)"

    if errorlevel 1 (
        echo.
        echo FAIL: Coverage is below threshold %THRESHOLD%%%
        exit /b 1
    ) else (
        echo.
        echo PASS: Coverage meets threshold %THRESHOLD%%%
    )
)

echo.
echo ==========================================
echo Tests completed successfully!
echo ==========================================
echo.
echo Coverage reports generated:
echo   - HTML: coverage_html_report\index.html
echo   - XML: coverage.xml
echo.
