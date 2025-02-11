#!/bin/bash

# Check Python version
python --version

# Install dependencies
pip install -r requirements.txt

# Run technical analysis tests
echo "Running technical analysis tests..."
python -m src.technical_analysis

# Run main program tests
echo "Running main program tests..."
python -m src.main --test

# If all tests pass, start the application
if [ $? -eq 0 ]; then
    echo "All tests passed. Starting application..."
    python -m src.main
else
    echo "Tests failed. Please check the logs."
    exit 1
fi 