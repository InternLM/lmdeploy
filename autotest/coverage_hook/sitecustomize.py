"""Enable coverage in CLI subprocesses when COVERAGE_PROCESS_START is set."""

try:
    import coverage
    coverage.process_startup()
except Exception:
    pass
