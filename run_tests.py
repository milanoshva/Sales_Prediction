import unittest
import sys
import os

# This script should be run from the project root directory.

# Get the absolute path to the project root directory.
project_root = os.path.abspath(os.path.dirname(__file__))

# Add the project root to the Python path.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if __name__ == '__main__':
    # Use the default test loader.
    loader = unittest.TestLoader()

    # Discover tests in the 'tests' directory.
    suite = loader.discover(start_dir='tests', pattern='test_*.py')

    # Create a test runner with verbosity for more detailed output.
    runner = unittest.TextTestRunner(verbosity=2)

    # Run the tests.
    result = runner.run(suite)

    # Exit with a status code indicating success (0) or failure (1).
    if result.wasSuccessful():
        sys.exit(0)
    else:
        sys.exit(1)
