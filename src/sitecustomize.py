"""Start coverage in Python subprocesses when the test runner requests it.

``pytest-cov`` measures the pytest process, but the nsys-ai CLI contract is
tested mostly through ``subprocess.run``. Python imports ``sitecustomize``
before application code, which gives those child interpreters a stable hook
for ``coverage.process_startup()``. The environment variable is installed by
``tests/conftest.py`` only for a ``pytest --cov`` run, so normal CLI invocations
do not import or start coverage as a side effect.
"""

import os

if os.environ.get("COVERAGE_PROCESS_START"):
    try:
        import coverage
    except ImportError:
        # A normal installed nsys-ai process may inherit the variable from a
        # parent shell without having the development extra installed. It
        # should still run; coverage is a test tool, not a runtime dependency.
        pass
    else:
        source = os.environ.get("COVERAGE_SOURCE")
        if source:
            # The child may change cwd before it starts (many CLI tests use a
            # temporary working directory). Override the config's relative
            # source with the absolute checkout path so coverage does not
            # silently record an empty run.
            coverage.Coverage(
                config_file=os.environ["COVERAGE_PROCESS_START"],
                source=[source],
                auto_data=True,
            ).start()
        else:
            coverage.process_startup()
