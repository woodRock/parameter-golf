#!/bin/bash

# Golf Caddy Installer
# Automates the setup of the 'caddy' experiment manager

set -e

echo "⛳ Installing Golf Caddy..."

# 1. Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed. Please install it and try again."
    exit 1
fi

# 2. Install dependencies
echo "📦 Installing dependencies (rich)..."
python3 -m pip install rich --quiet

# 3. Create bin directory if it doesn't exist
BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"

# 4. Create the symbolic link
SCRIPT_PATH="$(pwd)/caddy.py"
chmod +x "$SCRIPT_PATH"
ln -sf "$SCRIPT_PATH" "$BIN_DIR/caddy"

echo "✅ Created symbolic link: $BIN_DIR/caddy -> $SCRIPT_PATH"

# 5. Check if BIN_DIR is in PATH
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo "⚠️  Warning: $BIN_DIR is not in your PATH."
    echo "Add the following line to your ~/.bashrc or ~/.zshrc:"
    echo "export PATH=\"\$HOME/.local/bin:\$PATH\""
    
    # Optional: Automatically try to add it to .bashrc if the user confirms
    read -p "Would you like to add this to your ~/.bashrc now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "\n# Added by Golf Caddy Installer\nexport PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$HOME/.bashrc"
        echo "✅ Added to ~/.bashrc. Please run 'source ~/.bashrc' or restart your terminal."
    fi
else
    echo "✨ $BIN_DIR is already in your PATH."
fi

echo "🏁 Installation complete! You can now run 'caddy' from anywhere."
