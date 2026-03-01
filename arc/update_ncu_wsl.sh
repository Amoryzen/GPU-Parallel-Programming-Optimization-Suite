#!/bin/bash
set -e

# Configuration
NVIDIA_DIR="/mnt/c/Program Files/NVIDIA Corporation"
DEST_DIR="/opt/ncu_latest"
BIN_LINK="/usr/local/bin/ncu"

echo "Looking for Nsight Compute installations in '$NVIDIA_DIR'..."

# Find the latest version directory
LATEST_VERSION_DIR=$(ls -d "$NVIDIA_DIR/Nsight Compute "* 2>/dev/null | sort -V | tail -n 1)

if [ -z "$LATEST_VERSION_DIR" ]; then
    echo "Error: No Nsight Compute installation found in $NVIDIA_DIR"
    exit 1
fi

echo "Found latest version: $(basename "$LATEST_VERSION_DIR")"

# Prepare destination
echo "Preparing destination directory: $DEST_DIR"
if [ -d "$DEST_DIR" ]; then
    echo "Removing existing installation at $DEST_DIR..."
    rm -rf "$DEST_DIR"
fi
mkdir -p "$DEST_DIR"

# Copy Sections
echo "Copying sections..."
cp -r "$LATEST_VERSION_DIR/sections" "$DEST_DIR/"

# Copy Linux Binaries (from target/linux-desktop...)
LINUX_TARGET_SRC=$(ls -d "$LATEST_VERSION_DIR/target/linux-desktop-glibc_"* 2>/dev/null | head -n 1)
if [ -z "$LINUX_TARGET_SRC" ]; then
    echo "Error: Linux target binaries not found in $LATEST_VERSION_DIR/target/"
    exit 1
fi
echo "Copying binaries from $(basename "$LINUX_TARGET_SRC")..."
mkdir -p "$DEST_DIR/target"
cp -r "$LINUX_TARGET_SRC" "$DEST_DIR/target/"

# Copy Rules (from target/windows-desktop...)
WINDOWS_TARGET_SRC=$(ls -d "$LATEST_VERSION_DIR/target/windows-desktop"* 2>/dev/null | head -n 1)
if [ -z "$WINDOWS_TARGET_SRC" ]; then
    echo "Warning: Windows target (for rules) not found. Creating empty rules directory."
    mkdir -p "$DEST_DIR/rules"
else
    if [ -d "$WINDOWS_TARGET_SRC/rules" ]; then
        echo "Copying rules from $(basename "$WINDOWS_TARGET_SRC")..."
        cp -r "$WINDOWS_TARGET_SRC/rules" "$DEST_DIR/"
    else
        echo "Warning: Rules directory not found in $(basename "$WINDOWS_TARGET_SRC"). Creating empty rules directory."
        mkdir -p "$DEST_DIR/rules"
    fi
fi

# Set Permissions
echo "Setting permissions..."
chmod -R a+r "$DEST_DIR"
# Find the ncu binary within the copied target directory
NCU_BIN=$(find "$DEST_DIR/target" -name ncu -type f | head -n 1)
if [ -z "$NCU_BIN" ]; then
    echo "Error: ncu binary not found in $DEST_DIR/target"
    exit 1
fi
chmod a+x "$NCU_BIN"
chmod a+x "$(dirname "$NCU_BIN")"/*.sh 2>/dev/null || true

# Update Symlink
echo "Updating symlink '$BIN_LINK' -> '$NCU_BIN'..."
ln -sf "$NCU_BIN" "$BIN_LINK"

echo "--------------------------------------------------"
echo "Update Complete!"
$BIN_LINK --version
echo "--------------------------------------------------"
