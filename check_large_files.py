#!/usr/bin/env python3
import os
from pathlib import Path

def get_file_size_mb(filepath):
    """Get file size in MB"""
    return os.path.getsize(filepath) / (1024 * 1024)

def find_large_files(root_dir='.', threshold_mb=5):
    """Find all files larger than threshold"""
    large_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip hidden directories and common exclusions
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env']]

        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                size_mb = get_file_size_mb(filepath)
                if size_mb > threshold_mb:
                    large_files.append((filepath, size_mb))
            except (OSError, IOError):
                pass

    return sorted(large_files, key=lambda x: x[1], reverse=True)

if __name__ == '__main__':
    print(f"Scanning for files > 5 MB...\n")
    large_files = find_large_files()

    if not large_files:
        print("✓ No files larger than 5 MB found")
    else:
        print(f"Found {len(large_files)} large file(s):\n")
        total_size = 0
        for filepath, size_mb in large_files:
            print(f"  {filepath:<50} {size_mb:>8.2f} MB")
            total_size += size_mb
        print(f"\nTotal: {total_size:.2f} MB")
