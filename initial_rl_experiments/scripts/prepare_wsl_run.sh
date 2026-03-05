#!/bin/bash
# Copies the Luanti binary to a native WSL filesystem location
# so that executable permissions work correctly.
#
# Run this once before running tests:
#   bash scripts/prepare_wsl_run.sh

set -e

PROJECT_ROOT="/mnt/c/Users/marcu/OneDrive/Documente/GitHub/WiredTogether"
LUANTI_SRC="$PROJECT_ROOT/craftium/craftium/luanti"
LUANTI_DEST="$HOME/.wiredtogether/luanti"

echo "Preparing Luanti for WSL execution..."
echo "Source:      $LUANTI_SRC"
echo "Destination: $LUANTI_DEST"
echo ""

# Create destination directory
mkdir -p "$LUANTI_DEST"

# Copy Luanti files to native WSL filesystem
echo "Copying Luanti files (this may take a minute for the ~190MB binary)..."
rsync -av --delete "$LUANTI_SRC/" "$LUANTI_DEST/" || cp -r "$LUANTI_SRC/." "$LUANTI_DEST/"

# Make binary executable
chmod +x "$LUANTI_DEST/bin/luanti"
echo "Set executable permission on $LUANTI_DEST/bin/luanti"

# Verify
if [ -x "$LUANTI_DEST/bin/luanti" ]; then
    echo ""
    echo "✓ Luanti binary is now executable!"
    echo "  Path: $LUANTI_DEST/bin/luanti"
else
    echo "ERROR: Failed to make binary executable!"
    exit 1
fi

echo ""
echo "Done! Now set CRAFTIUM_LUANTI_DIR in your environment:"
echo "  export CRAFTIUM_LUANTI_DIR=$LUANTI_DEST"
echo ""
echo "Or run directly with:"
echo "  CRAFTIUM_LUANTI_DIR=$LUANTI_DEST python scripts/test_openworld.py"
