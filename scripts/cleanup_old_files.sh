#!/bin/bash

echo "🧹 Cleaning up old complex implementation..."
echo "============================================"

# List of directories and files to remove
OLD_DIRS=(
    "faster_whisper_daemon"
    "benchmark_results" 
    "outputs"
    "tests"
    "tools"
    "cli"
    "config"
    "audio_processing"
    "api_integrations"
    "post_processing"
)

OLD_FILES=(
    "requirements-local.txt"
    "CLEANUP_SUMMARY.md"
)

# Remove old directories
for dir in "${OLD_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "🗑️  Removing directory: $dir"
        rm -rf "$dir"
    fi
done

# Remove old files  
for file in "${OLD_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "🗑️  Removing file: $file"
        rm "$file"
    fi
done

# Keep the original README as backup
if [ -f "README.md" ]; then
    echo "📄 Backing up original README.md to README_ORIGINAL.md"
    mv README.md README_ORIGINAL.md
fi

# Use the minimal README as the main one
if [ -f "README_MINIMAL.md" ]; then
    echo "📄 Setting up minimal README as main README"
    mv README_MINIMAL.md README.md
fi

echo ""
echo "✅ Cleanup complete!"
echo "📁 Preserved assets are in internal/abandoned/future_features/"
echo "🎯 Core functionality is now in voice_transcriber.py"