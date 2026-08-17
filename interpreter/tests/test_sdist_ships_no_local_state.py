"""Acceptance: the published sdist carries no local development state.

A source distribution is the package's public form. Hypothesis keeps a
per-machine example cache under ``.hypothesis/``; shipping it publishes
whatever that cache absorbed, and any absolute paths it may carry. The
archive must contain no ``.hypothesis`` member, including one seeded with
a private path immediately before the build.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path, PurePosixPath

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
CAPSULE_ROOT = REPOSITORY_ROOT / "abstraction-capsule"
ROOT_LICENSE = REPOSITORY_ROOT / "LICENSE"
EXPECTED_INTERPRETER_REQUIREMENT = "rs-metalang-ref==0.1.0"
# Assembled at runtime so the literal never appears in this file (the sdist
# ships tests/, and a literal here would make the byte scan find itself).
PRIVATE_MARKER = "/".join(("", "Users", "private-marker-home", "secret-project"))  # noqa: FLY002 - see comment above


def _build_sdist(source_root: Path, out_dir: Path) -> Path:
    uv = shutil.which("uv")
    assert uv, "the uv executable must be on PATH to build the sdist under test"
    subprocess.run(
        [uv, "build", "--sdist", "--out-dir", str(out_dir)],
        cwd=source_root,
        check=True,
        capture_output=True,
    )
    archives = sorted(out_dir.glob("*.tar.gz"))
    assert archives, "uv build --sdist produced no archive"
    return archives[-1]


def _build_publication_archives(source_root: Path, out_dir: Path) -> tuple[Path, Path]:
    uv = shutil.which("uv")
    assert uv, "the uv executable must be on PATH to build publication archives"
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            uv,
            "build",
            "--wheel",
            "--sdist",
            "--no-sources",
            "--out-dir",
            str(out_dir),
        ],
        cwd=source_root,
        check=True,
        capture_output=True,
    )
    wheels = sorted(out_dir.glob("*.whl"))
    sdists = sorted(out_dir.glob("*.tar.gz"))
    assert len(wheels) == 1, f"expected one wheel, found {wheels}"
    assert len(sdists) == 1, f"expected one sdist, found {sdists}"
    return wheels[0], sdists[0]


def _member_names(archive: Path) -> list[str]:
    with tarfile.open(archive) as tar:
        return tar.getnames()


def _assert_intended_members(names: list[str]) -> None:
    """A cache-free archive proves nothing if it is also content-free."""
    for suffix in (
        "/pyproject.toml",
        "/README.md",
        "/src/rs_metalang_ref/__init__.py",
    ):
        assert any(name.endswith(suffix) for name in names), (
            f"sdist is missing an intended member ending in {suffix!r}"
        )


def test_sdist_contains_no_hypothesis_members(tmp_path):
    """The archive built from the package as-is ships no .hypothesis member."""
    archive = _build_sdist(PACKAGE_ROOT, tmp_path)

    names = _member_names(archive)
    _assert_intended_members(names)
    offending = [name for name in names if ".hypothesis" in name]
    assert not offending, (
        "sdist ships local Hypothesis state; the archive is the public form "
        f"of the package (members: {offending[:5]}"
        f"{'...' if len(offending) > 5 else ''})"
    )


def test_sdist_excludes_a_freshly_seeded_private_path_cache_entry(tmp_path):
    """A cache entry carrying a private absolute path never reaches the archive.

    Works on an isolated copy so the real package tree is never written to.
    """
    isolated = tmp_path / "isolated"
    shutil.copytree(
        PACKAGE_ROOT,
        isolated,
        # .gitignore is omitted deliberately: the exclusion under test must come
        # from the package's own sdist contract (pyproject), not from Hatch
        # falling back to VCS-ignore semantics.
        ignore=shutil.ignore_patterns(
            ".hypothesis", "__pycache__", "*.egg-info", "dist", ".ruff_cache",
            ".gitignore",
        ),
    )
    seed = isolated / ".hypothesis" / "constants" / "seeded-entry"
    seed.parent.mkdir(parents=True)
    seed.write_text(f"marker = {PRIVATE_MARKER!r}\n", encoding="utf-8")

    archive = _build_sdist(isolated, tmp_path / "out")

    names = _member_names(archive)
    _assert_intended_members(names)
    assert not [n for n in names if ".hypothesis" in n], (
        "the seeded cache directory reached the archive"
    )
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            handle = tar.extractfile(member)
            if handle and PRIVATE_MARKER.encode() in handle.read():
                pytest.fail(f"private path bytes reached the archive in {member.name}")


@pytest.mark.parametrize(
    "source_root",
    [PACKAGE_ROOT, CAPSULE_ROOT],
    ids=["interpreter", "abstraction-capsule"],
)
def test_publication_archives_include_root_license(source_root, tmp_path):
    """Each publication archive carries the repository's MIT licence text."""
    wheel, sdist = _build_publication_archives(source_root, tmp_path)
    expected = ROOT_LICENSE.read_bytes()
    defects: list[str] = []

    with zipfile.ZipFile(wheel) as archive:
        wheel_members = [
            name
            for name in archive.namelist()
            if name.endswith(".dist-info/licenses/LICENSE")
        ]
        if len(wheel_members) != 1:
            defects.append(
                f"{wheel.name} has {len(wheel_members)} conventional LICENSE members"
            )
        elif archive.read(wheel_members[0]) != expected:
            defects.append(f"{wheel.name} does not carry the root LICENSE bytes")

    with tarfile.open(sdist) as archive:
        sdist_members = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and PurePosixPath(member.name).name == "LICENSE"
            and len(PurePosixPath(member.name).parts) == 2
        ]
        if len(sdist_members) != 1:
            defects.append(
                f"{sdist.name} has {len(sdist_members)} top-level LICENSE members"
            )
        else:
            handle = archive.extractfile(sdist_members[0])
            assert handle is not None
            if handle.read() != expected:
                defects.append(f"{sdist.name} does not carry the root LICENSE bytes")

    assert not defects, "; ".join(defects)


def _requires_dist(metadata: bytes) -> set[str]:
    message = BytesParser(policy=default).parsebytes(metadata)
    return {
        "".join(requirement.split()).lower().replace("_", "-")
        for requirement in message.get_all("Requires-Dist", [])
    }


def test_capsule_archives_pin_the_interpreter_dependency(tmp_path):
    """The capsule wheel and sdist select the matching interpreter release."""
    wheel, sdist = _build_publication_archives(CAPSULE_ROOT, tmp_path)
    defects: list[str] = []

    with zipfile.ZipFile(wheel) as archive:
        metadata_members = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        assert len(metadata_members) == 1, metadata_members
        requirements = _requires_dist(archive.read(metadata_members[0]))
        if EXPECTED_INTERPRETER_REQUIREMENT not in requirements:
            defects.append(
                f"{wheel.name} requirements are {sorted(requirements)!r}"
            )

    with tarfile.open(sdist) as archive:
        metadata_members = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and PurePosixPath(member.name).name == "PKG-INFO"
            and len(PurePosixPath(member.name).parts) == 2
        ]
        assert len(metadata_members) == 1, metadata_members
        handle = archive.extractfile(metadata_members[0])
        assert handle is not None
        requirements = _requires_dist(handle.read())
        if EXPECTED_INTERPRETER_REQUIREMENT not in requirements:
            defects.append(
                f"{sdist.name} requirements are {sorted(requirements)!r}"
            )

    assert not defects, (
        f"capsule archives must require {EXPECTED_INTERPRETER_REQUIREMENT}; "
        + "; ".join(defects)
    )


def test_publication_wheels_install_and_run_without_the_source_tree(tmp_path):
    """The paired wheels run without a registry or ``../interpreter`` path."""
    interpreter_wheel, _ = _build_publication_archives(
        PACKAGE_ROOT, tmp_path / "interpreter-dist"
    )
    capsule_wheel, _ = _build_publication_archives(
        CAPSULE_ROOT, tmp_path / "capsule-dist"
    )
    uv = shutil.which("uv")
    assert uv, "the uv executable must be on PATH to install publication wheels"

    environment = tmp_path / "environment"
    subprocess.run(
        [uv, "venv", "--python", sys.executable, str(environment)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    environment_python = environment / "bin" / "python"
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(environment_python),
            "pydantic==2.11.7",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(environment_python),
            "--no-index",
            str(interpreter_wheel),
            str(capsule_wheel),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    run_directory = tmp_path / "outside-source-tree"
    run_directory.mkdir()
    completed = subprocess.run(
        [environment_python, "-I", "-m", "rs_capsule"],
        cwd=run_directory,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    records = json.loads(completed.stdout)
    assert [record["card_id"] for record in records] == [
        "tail-window-1-v1",
        "obligation-table-v1",
    ]
