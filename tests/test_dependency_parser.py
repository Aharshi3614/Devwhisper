"""
Unit tests for dependency_parser module (Issue #219).
"""

import os
import tempfile
from dependency_parser import (
    parse_requirements_txt,
    parse_pyproject_toml,
    generate_dependency_summary,
)


def test_parse_requirements_txt():
    content = """
# Sample requirements file
fastapi==0.139.2
uvicorn>=0.51.0 # ASGI server
requests # HTTP client
-r base.txt
--extra-index-url https://pypi.org/simple
"""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name

    try:
        deps = parse_requirements_txt(path)
        assert "fastapi" in deps
        assert "uvicorn" in deps
        assert "requests" in deps
        assert len(deps) == 3
    finally:
        os.unlink(path)


def test_parse_pyproject_toml():
    content = """
[project]
name = "demo"
version = "0.1.0"
dependencies = [
    "flask>=2.0.0",
    "pydantic",
]

[tool.poetry.dependencies]
python = "^3.10"
numpy = "^1.21.0"
"""
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write(content)
        path = f.name

    try:
        deps = parse_pyproject_toml(path)
        assert "flask" in deps
        assert "pydantic" in deps
        assert "numpy" in deps
        assert "python" not in deps
    finally:
        os.unlink(path)


def test_generate_dependency_summary():
    with tempfile.TemporaryDirectory() as tmpdir:
        req_path = os.path.join(tmpdir, "requirements.txt")
        pyproject_path = os.path.join(tmpdir, "pyproject.toml")

        with open(req_path, "w") as f:
            f.write("fastapi==0.100.0\npytest\n")

        with open(pyproject_path, "w") as f:
            f.write('[project]\ndependencies = ["fastapi", "black"]\n')

        summary = generate_dependency_summary(tmpdir)
        assert summary["total_unique_dependencies"] == 3
        assert sorted(summary["all_dependencies"]) == ["black", "fastapi", "pytest"]
        assert "fastapi" in summary["requirements_txt"]
        assert "black" in summary["pyproject_toml"]
