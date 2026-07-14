#!/bin/bash
# Test runner script with coverage for LMDeploy
# Usage: ./scripts/run_tests.sh [options]

set -e

# Default values
COVERAGE=${COVERAGE:-1}
THRESHOLD=${THRESHOLD:-40}
TEST_PATH=${TEST_PATH:-"tests"}
PARALLEL=${PARALLEL:-1}
NUM_WORKERS=${NUM_WORKERS:-4}

echo "=========================================="
echo "LMDeploy Test Runner with Coverage"
echo "=========================================="
echo ""

# Build pytest command
PYTEST_CMD="pytest"

# Add coverage options if enabled
if [ "$COVERAGE" = "1" ]; then
    echo "Coverage reporting: ENABLED"
    echo "Coverage threshold: ${THRESHOLD}%"
    PYTEST_CMD="$PYTEST_CMD --cov=lmdeploy --cov-report=term-missing --cov-report=html --cov-report=xml"
else
    echo "Coverage reporting: DISABLED"
fi

# Add parallel execution if enabled
if [ "$PARALLEL" = "1" ]; then
    echo "Parallel execution: ENABLED (${NUM_WORKERS} workers)"
    PYTEST_CMD="$PYTEST_CMD -n ${NUM_WORKERS}"
else
    echo "Parallel execution: DISABLED"
fi

# Add test path and any additional arguments
PYTEST_CMD="$PYTEST_CMD $TEST_PATH ${@}"

echo ""
echo "Running: $PYTEST_CMD"
echo "=========================================="
echo ""

# Run tests
eval $PYTEST_CMD

# Check coverage threshold if coverage is enabled
if [ "$COVERAGE" = "1" ]; then
    echo ""
    echo "=========================================="
    echo "Checking coverage threshold..."

    # Get total coverage from coverage report
    TOTAL_COVERAGE=$(python -c "import coverage; c = coverage.Coverage(); c.load(); print(f'{c.report():.1f}')")

    echo "Total coverage: ${TOTAL_COVERAGE}%"
    echo "Required threshold: ${THRESHOLD}%"

    # Compare coverage with threshold
    if (( $(echo "$TOTAL_COVERAGE < $THRESHOLD" | bc -l) )); then
        echo ""
        echo "❌ FAIL: Coverage ${TOTAL_COVERAGE}% is below threshold ${THRESHOLD}%"
        exit 1
    else
        echo ""
        echo "✅ PASS: Coverage ${TOTAL_COVERAGE}% meets threshold ${THRESHOLD}%"
    fi
fi

echo ""
echo "=========================================="
echo "Tests completed successfully!"
echo "=========================================="
echo ""
echo "Coverage reports generated:"
echo "  - HTML: coverage_html_report/index.html"
echo "  - XML: coverage.xml"
echo ""
