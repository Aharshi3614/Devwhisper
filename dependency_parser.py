"""
dependency_parser.py — Repository Dependency Parser and Summarizer for DevWhisper.

Parses dependency manifest files (requirements.txt, pyproject.toml) to summarize
project dependencies after codebase indexing.
"""

import os
import re
from typing import Dict, List, Any
from logger import logger

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None  # Fallback manual parsing if tomllib is not available


def parse_requirements_txt(filepath: str) -> List[str]:
    """
    Parse a requirements.txt file and return a list of cleaned dependency package names.
    Ignores comments, empty lines, options (like -r, -e, --index-url), and version specifiers.
    """
    if not os.path.exists(filepath):
        return []

    dependencies = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                # Skip comments, empty lines, and pip flags
                if not line or line.startswith("#") or line.startswith("-"):
                    continue
                # Remove inline comments
                if " #" in line:
                    line = line.split(" #")[0].strip()

                # Extract package name before specifiers/extras/environment markers
                match = re.match(r"^([a-zA-Z0-9_\-\.]+)", line)
                if match:
                    pkg_name = match.group(1)
                    if pkg_name not in dependencies:
                        dependencies.append(pkg_name)
    except Exception as e:
        logger.warning(f"Error parsing requirements.txt at {filepath}: {e}")

    return sorted(dependencies)


def parse_pyproject_toml(filepath: str) -> List[str]:
    """
    Parse a pyproject.toml file where available and return a list of dependency package names.
    Supports standard PEP 621 [project.dependencies] and Poetry [tool.poetry.dependencies].
    """
    if not os.path.exists(filepath):
        return []

    dependencies = set()
    try:
        if tomllib:
            with open(filepath, "rb") as f:
                data = tomllib.load(f)

            # PEP 621 dependencies
            proj_deps = data.get("project", {}).get("dependencies", [])
            if isinstance(proj_deps, list):
                for dep in proj_deps:
                    match = re.match(r"^([a-zA-Z0-9_\-\.]+)", str(dep).strip())
                    if match:
                        dependencies.add(match.group(1))

            # Poetry dependencies
            poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
            if isinstance(poetry_deps, dict):
                for pkg in poetry_deps.keys():
                    if pkg.lower() != "python":
                        dependencies.add(pkg)

            # Flit / hatch / pdm standard optional/main dependencies
            opt_deps = data.get("project", {}).get("optional-dependencies", {})
            if isinstance(opt_deps, dict):
                for group, deps in opt_deps.items():
                    if isinstance(deps, list):
                        for dep in deps:
                            match = re.match(r"^([a-zA-Z0-9_\-\.]+)", str(dep).strip())
                            if match:
                                dependencies.add(match.group(1))
        else:
            # Fallback simple regex extraction if tomllib module is unavailable
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                # Basic match for requirements in dependencies list
                matches = re.findall(r'"([a-zA-Z0-9_\-\.]+)[^"]*"', content)
                for m in matches:
                    if m.lower() not in ("python", "setuptools", "wheel", "flit_core", "hatchling"):
                        dependencies.add(m)
    except Exception as e:
        logger.warning(f"Error parsing pyproject.toml at {filepath}: {e}")

    return sorted(list(dependencies))


def generate_dependency_summary(directory: str) -> Dict[str, Any]:
    """
    Analyze supported dependency files in the given directory and generate a summary.

    Returns dict containing:
      - requirements_txt: list of detected requirements
      - pyproject_toml: list of detected pyproject dependencies
      - total_unique_dependencies: count of unique dependencies across all files
      - all_dependencies: list of sorted unique dependency names
    """
    req_file = os.path.join(directory, "requirements.txt")
    pyproject_file = os.path.join(directory, "pyproject.toml")

    req_deps = parse_requirements_txt(req_file)
    pyproject_deps = parse_pyproject_toml(pyproject_file)

    all_deps = sorted(list(set(req_deps) | set(pyproject_deps)))

    summary = {
        "requirements_txt": req_deps,
        "pyproject_toml": pyproject_deps,
        "total_unique_dependencies": len(all_deps),
        "all_dependencies": all_deps,
    }
    return summary
