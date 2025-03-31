#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
# This mimics the behavior of using && between commands.
set -e

echo "--- Starting RooFlow config setup ---"

# Clone the repository
echo "Cloning RooFlow repository..."
git clone https://github.com/GreatScottyMac/RooFlow

# Move config files (including hidden ones)
echo "Moving config files..."
shopt -s dotglob  # Enable matching hidden files (like .gitignore)
mv RooFlow/config/* ./
shopt -u dotglob  # Disable matching hidden files again (good practice)

# Copy .clinerules-default file
echo "Copying .clinerules-default..."
cp RooFlow/config/default-mode/.clinerules-default ./ || echo "Warning: Failed to copy .clinerules-default. Manual copy might be needed."

# Make the script executable
echo "Setting permissions for insert-variables.sh..."
chmod +x insert-variables.sh

# Clean up unnecessary files and directories
echo "Cleaning up..."
rm -f insert-variables.cmd   # -f ignores error if file doesn't exist
rm -rf default-mode          # -rf removes directories recursively and ignores errors
rm -rf RooFlow               # Remove the cloned repo directory

# Run the setup script
echo "Running insert-variables.sh..."
./insert-variables.sh
echo "insert-variables.sh completed successfully. Removing it..."
rm -f insert-variables.sh

echo "Scheduling self-deletion of install_rooflow.sh..."
( sleep 1 && rm -f "$0" ) &
echo "--- RooFlow config setup complete ---"