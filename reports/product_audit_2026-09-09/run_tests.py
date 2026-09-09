"""Record the final repository unittest result and full log for this audit."""
from pathlib import Path
import argparse
import json
import sys
import time
import unittest

def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    output = Path(__file__).parent
    start = time.perf_counter()
    with (output / "unittest.log").open("w") as log:
        original_out, original_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = log
        try:
            suite = unittest.defaultTestLoader.discover(str(root / "tests"))
            result = unittest.TextTestRunner(stream=log, verbosity=1).run(suite)
        finally:
            sys.stdout, sys.stderr = original_out, original_err
    payload = {
        "command": ".venv/bin/python reports/product_audit_2026-09-09/run_tests.py",
        "suite": "tests", "tests_run": result.testsRun,
        "failures": [str(case) for case, _ in result.failures],
        "errors": [str(case) for case, _ in result.errors],
        "skipped": [{"test": str(case), "reason": reason} for case, reason in result.skipped],
        "seconds": round(time.perf_counter() - start, 3), "ok": result.wasSuccessful(),
    }
    (output / "tests.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
