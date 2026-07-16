#!/usr/bin/env python3
"""
Validation script for the sample codebase directory before indexing.
Performs pre-flight checks to detect common issues early.
"""

import os
import sys

# NOTE: Keep this in sync with indexer.py SUPPORTED_EXTENSIONS
SUPPORTED_EXTENSIONS = {".py"}

IGNORED_DIRS = {"__pycache__", ".git", ".mypy_cache", ".pytest_cache", ".venv", "venv", "node_modules"}


def should_ignore_dir(dir_name):
    """Check if a directory should be ignored."""
    return dir_name in IGNORED_DIRS or dir_name.startswith(".")


def validate_codebase(directory):
    """
    Validate a codebase directory for indexing.

    Returns:
        tuple: (valid_files, skipped_files, warnings, errors)
    """
    valid_files = []
    skipped_files = []
    warnings = []
    errors = []

    # Check if directory exists
    if not os.path.exists(directory):
        errors.append(f"Directory does not exist: {directory}")
        return valid_files, skipped_files, warnings, errors

    if not os.path.isdir(directory):
        errors.append(f"Path is not a directory: {directory}")
        return valid_files, skipped_files, warnings, errors

    # Walk the directory and collect all files
    found_files = []
    unsupported_files = []

    for root, dirs, files in os.walk(directory):
        # Filter out directories we want to skip
        dirs[:] = [d for d in dirs if not should_ignore_dir(d)]

        for file in files:
            # Skip hidden files
            if file.startswith("."):
                continue
            path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                found_files.append(path)
            else:
                unsupported_files.append(path)

    # Check if directory is empty
    if not found_files and not unsupported_files:
        errors.append("Directory is empty")
        return valid_files, skipped_files, warnings, errors

    # Report unsupported files
    if unsupported_files:
        warnings.append("Skipped:")
        for path in unsupported_files:
            warnings.append(f"  {path}")

    # Check that supported source files are present
    if not found_files:
        errors.append(f"No supported source files found (supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))})")
        return valid_files, skipped_files, warnings, errors

    # Validate each supported file
    for filepath in found_files:
        try:
            # Check if file is empty
            if os.path.getsize(filepath) == 0:
                skipped_files.append((filepath, "File is empty"))
                continue

            # Try to read the file to check if it's readable
            with open(filepath, "r", encoding="utf-8") as f:
                f.read()

            # If we got here, the file is valid
            valid_files.append(filepath)

        except PermissionError:
            skipped_files.append((filepath, "Permission denied"))
        except UnicodeDecodeError:
            skipped_files.append((filepath, "Invalid text encoding"))
        except OSError as e:
            errors.append(f"Error reading file {filepath}: {e}")
        except Exception as e:
            errors.append(f"Unexpected error with file {filepath}: {e}")

    return valid_files, skipped_files, warnings, errors


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "./sample_codebase"

    print(f"Validating codebase: {target}\n")

    valid_files, skipped_files, warnings, errors = validate_codebase(target)

    # Print errors
    if errors:
        print("[ERROR] Errors:")
        for error in errors:
            print(f"  [ERROR] {error}")

    # Print skipped files
    if skipped_files:
        print(f"\n[WARNING] Skipped files: {len(skipped_files)}")
        for filepath, reason in skipped_files:
            print(f"  [WARNING] {filepath} ({reason})")

    # Print warnings
    if warnings:
        for warning in warnings:
            print(f"\n[WARNING] {warning}")

    # Summary
    print("\nValidation Summary")
    print("-" * 20)
    print(f"Valid: {len(valid_files)}")
    print(f"Skipped: {len(skipped_files)}")
    print(f"Errors: {len(errors)}")

    # Determine exit code
    if errors:
        sys.exit(1)
    elif not valid_files:
        # Nothing valid to index
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()