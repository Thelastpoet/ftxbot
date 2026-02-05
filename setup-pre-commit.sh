#!/bin/bash
# setup-pre-commit.sh
# Script to set up pre-commit hooks for the Forex Trading Bot project

echo "Setting up pre-commit hooks for Forex Trading Bot..."

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "pre-commit is not installed. Installing..."
    pip install pre-commit
fi

# Install the git hooks
echo "Installing git hooks..."
pre-commit install

# Run on all files to verify setup
echo "Running pre-commit on all files to verify setup..."
pre-commit run --all-files

echo "Pre-commit hooks have been set up successfully!"
echo ""
echo "The following hooks are now active:"
echo "- Black: Formats Python code"
echo "- Flake8: Lints Python code"
echo "- Isort: Sorts imports"
echo "- JSON/YAML validation"
echo "- End of file fixer"
echo "- Trailing whitespace removal"
echo "- Debug statement checker"
echo ""
echo "To run pre-commit manually on all files, use:"
echo "  pre-commit run --all-files"
echo ""
echo "To run pre-commit on staged files only, use:"
echo "  pre-commit run"