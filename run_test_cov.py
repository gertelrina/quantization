import os
import traceback
import unittest
from pathlib import Path

import coverage

def test_suite(tests_root_dir: Path):
    loader = unittest.TestLoader()

    suite = unittest.TestSuite()

    suite.addTest(loader.discover(str(tests_root_dir)))

    return suite


if __name__ == "__main__":
    cov = coverage.Coverage(omit=["*/venv*/","*/env*/", "*Library*"])

    cov.start()

    script_path = Path(__file__).parent
    tests_root_dir = script_path / "test"

    try:
        os.chdir(tests_root_dir.as_posix())
        runner = unittest.TextTestRunner()
        runner.run(test_suite(tests_root_dir))
        
    except Exception:
        traceback.print_exc()
        print("Error running test")
    finally:
        os.chdir(script_path.as_posix())

    cov.stop()

    cov.report()  # prints simple report to console
    # xml_report_path = "coverage-reports/coverage-all.xml"
    # cov.xml_report(outfile=xml_report_path)
    # print(f"Generated xml report at {Path(xml_report_path).resolve()}")
