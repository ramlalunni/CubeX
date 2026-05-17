#!/bin/bash
# CubeX Installation Script
# Supports Linux (Debian/Ubuntu) and macOS

set -e # Exit immediately if a command exits with a non-zero status.

echo "======================================"
echo "    Welcome to the CubeX Installer"
echo "======================================"

# 1. OS Detection and System Dependencies
OS="$(uname -s)"
case "${OS}" in
    Linux*)
        echo "[*] Detected OS: Linux"
        # Check for apt-get (Debian/Ubuntu)
        if command -v apt-get >/dev/null; then
            echo "[*] Installing required fonts (fonts-noto-color-emoji) for UI emojis..."
            # Running with sudo might prompt the user for their password
            sudo apt-get update -qq
            sudo apt-get install -y fonts-noto-color-emoji
        else
            echo "[!] 'apt-get' not found. If you are on a non-Debian/Ubuntu system, please ensure an emoji font package is installed manually."
        fi
        ;;
    Darwin*)
        echo "[*] Detected OS: macOS"
        echo "[*] macOS natively supports emojis (Apple Color Emoji). Skipping font installation."
        ;;
    *)
        echo "[!] Unsupported OS: ${OS}. Proceeding with Python dependencies only."
        ;;
esac

echo ""

# 2. Check for Python 3
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1 && python --version 2>&1 | grep -q "Python 3"; then
    PYTHON_CMD="python"
else
    echo "[-] Error: Python 3 is not installed or not found in PATH. Please install Python 3.8+ and try again."
    exit 1
fi
echo "[*] Using Python command: $PYTHON_CMD"

# 3. Environment Setup
echo -n "Do you want to install CubeX in a new virtual environment? (y/n) [y]: "
read -r CREATE_VENV
CREATE_VENV=${CREATE_VENV:-y}

if [[ "$CREATE_VENV" =~ ^[Yy]$ ]]; then
    echo -n "Enter the name for the new virtual environment [CubeX_venv]: "
    read -r VENV_DIR
    VENV_DIR=${VENV_DIR:-CubeX_venv}

    if [ ! -d "$VENV_DIR" ]; then
        echo "[*] Creating Python virtual environment in './$VENV_DIR'..."
        "$PYTHON_CMD" -m venv "$VENV_DIR"
    else
        echo "[*] Virtual environment './$VENV_DIR' already exists."
    fi

    # Activate Virtual Environment
    echo "[*] Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
else
    echo "[*] Proceeding with the current environment."
fi

# 4. Install Python Dependencies
echo "[*] Installing Python dependencies from requirements.txt..."
# Use "$PYTHON_CMD -m pip" which is safer than just "pip" when dealing with mixed environments
"$PYTHON_CMD" -m pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    "$PYTHON_CMD" -m pip install -r requirements.txt
else
    echo "[-] Error: requirements.txt not found!"
    exit 1
fi

# 5. Create CubeX Launcher script
echo "[*] Creating CubeX.sh launcher script..."
cat << EOF > CubeX.sh
#!/bin/bash
# CubeX Auto-Generated Launcher
EOF

if [[ "$CREATE_VENV" =~ ^[Yy]$ ]]; then
    echo "source \"\$PWD/$VENV_DIR/bin/activate\"" >> CubeX.sh
    echo "python \"\$PWD/main.py\" \"\$@\"" >> CubeX.sh
else
    echo "$PYTHON_CMD \"\$PWD/main.py\" \"\$@\"" >> CubeX.sh
fi
chmod +x CubeX.sh

echo "======================================"
echo "    CubeX Installation Complete! 🚀"
echo "======================================"
echo ""
echo "To run CubeX, simply double-click or run from the terminal:"
echo "  ./CubeX.sh"
echo ""
