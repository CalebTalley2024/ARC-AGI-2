#!/usr/bin/env python3
"""
Script to run all tests in the tests folder.
"""

import subprocess
import sys
from pathlib import Path


def run_all_tests():
    """Run all test files in the tests directory."""

    # Get the project root directory (parent of scripts directory)
    project_root = Path(__file__).parent.parent

    # Find all test files
    tests_dir = project_root / "tests"
    test_files = list(tests_dir.glob("test_*.py"))

    if not test_files:
        print("No test files found in tests/ directory")
        return False

    print(f"Found {len(test_files)} test files:")
    for test_file in test_files:
        print(f"  - {test_file.name}")
    print()

    # Run each test file
    all_passed = True

    for test_file in test_files:
        print(f"Running {test_file.name}...")
        print("-" * 50)

        try:
            # Run pytest on individual test file
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(test_file),
                    "-v",  # verbose output
                    "--tb=short",  # shorter traceback format
                ],
                cwd=project_root,
                capture_output=False,
            )

            if result.returncode == 0:
                print(f" {test_file.name} PASSED")
            else:
                print(f" {test_file.name} FAILED")
                all_passed = False

        except Exception as e:
            print(f" Error running {test_file.name}: {e}")
            all_passed = False

        print()

    # Summary
    print("=" * 50)
    if all_passed:
        print("ALL TESTS PASSED!")
        return True
    else:
        print(" SOME TESTS FAILED!")
        return False


def run_tests_together():
    """Run all tests together with pytest."""

    # Get the project root directory (parent of scripts directory)
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / "tests"

    print("Running all tests together...")
    print("-" * 50)

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(tests_dir),
                "-v",  # verbose output
                "--tb=short",  # shorter traceback format
            ],
            cwd=project_root,
            capture_output=False,
        )

        if result.returncode == 0:
            print("\n ALL TESTS PASSED!")
            return True
        else:
            print("\n SOME TESTS FAILED!")
            return False

    except Exception as e:
        print(f"Error running tests: {e}")
        return False


if __name__ == "__main__":
    print("ARC-AGI Test Runner")
    print("=" * 50)

    # Check if user wants individual or combined test runs
    if len(sys.argv) > 1 and sys.argv[1] == "--individual":
        success = run_all_tests()
    else:
        success = run_tests_together()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
