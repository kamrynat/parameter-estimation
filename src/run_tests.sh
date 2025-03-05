#!/bin/bash
#written with chatgpt
# Display a message indicating the tests are running
echo "Running tests for Parameter Estimation package..."

# Navigate to the project root directory
cd "$(dirname "$0")/.."

# Set PYTHONPATH to include the project root directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the tests using Python's unittest module
# Use python3 instead of python
python3 -m unittest discover -s tests -p "test_*.py" -v

# Check the exit status of the test run
if [ $? -eq 0 ]; then
    echo "All tests passed successfully!"
    exit 0
else
    echo "Some tests failed. Please check the output above for details."
    exit 1
fi


