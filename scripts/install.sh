#!/bin/bash
# KCake Installation Script

set -e

echo "========================================"
echo "  KCake Installation Script"
echo "========================================"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

if [[ ! "$PYTHON_VERSION" =~ ^3\.(10|11|12) ]]; then
    echo "Error: KCake requires Python 3.10, 3.11, or 3.12"
    exit 1
fi

# Create virtual environment (optional but recommended)
if [ "$1" == "--venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv kcake-env
    source kcake-env/bin/activate
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo "Installing KCake..."
pip install -e .

# Verify installation
echo ""
echo "Verifying installation..."
python3 -m kcake --version

echo ""
echo "========================================"
echo "  Installation Complete!"
echo "========================================"
echo ""
echo "Quick start:"
echo "  python -m kcake serve --model <model_name> --cluster-key <key>"
echo "  python -m kcake chat --model <model_name>"
echo ""
