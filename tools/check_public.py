"""Explicit public-file allowlist, SHA-256 seal and conservative disclosure scan.

No research workspace is imported. The manifest omits itself to avoid a cycle.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"
FILES = frozenset({
    ".gitignore", ".github/workflows/tests.yml", "README.md", "CITATION.cff", "LICENSE",
    "pyproject.toml", "requirements.txt", "benchmarks/__init__.py",
    "benchmarks/connected_grid_calibration.py",
    "benchmarks/local_connected_cuda_smoke.py",
    "benchmarks/run_connected_grid_matrix.py",
    "configs/connected_grid_reference.json",
    "docs/REPRODUCIBILITY.md", "docs/EUROHPC_PLAYGROUND_SCOPE.md",
    "docs/CLAIM_BOUNDARIES.md", "docs/SECURITY_AND_IP_SCOPE.md",
    "hpc/leonardo_smoke.sbatch", "hpc/leonardo_calibration_v2.sbatch",
    "results/README.md", "results/connected_grid_smoke_s1.json",
    "results/connected_grid_smoke_s2.json", "results/connected_grid_smoke_s3.json",
    "results/local_connected_grid_cuda_smoke.json",
    "results/playground_calibration_matrix_v2.csv",
    "tests/test_connected_grid_calibration.py", "tests/test_public_repository.py",
    "tools/check_public.py",
})
IGNORED_DIRS = frozenset({".git", ".npc-context", ".codex-npc", ".venv",
                          "__pycache__", ".pytest_cache", ".mypy_cache"})
FORBIDDEN_PATH_PARTS = frozenset({".env", "calibration", "evaluation", "cineldi",
                                  "task_a", "task_b", "promotion", "ledger"})
SECRET_PATTERNS = (
    re.compile(rb"(?i)github_pat_[A-Za-z0-9_]{16,}|gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(rb"(?i)[A-Z]:\\" + b"Users" + rb"\\|/" + b"Users" + rb"/[^/\s]+/|/" + b"home" + rb"/[^/\s]+/"),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def present_files() -> set[str]:
    result: set[str] = set()
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in IGNORED_DIRS or part.startswith(".venv-")
                                     or part.endswith(".egg-info")
                                     for part in path.relative_to(ROOT).parts):
            continue
        result.add(path.relative_to(ROOT).as_posix())
    return result


def scan() -> None:
    actual = present_files() - {"MANIFEST.sha256"}
    if actual != FILES:
        raise ValueError(f"Public allowlist mismatch: extra={sorted(actual - FILES)}, "
                         f"missing={sorted(FILES - actual)}")
    for rel in sorted(FILES):
        lowered = rel.lower()
        parts = set(Path(lowered).parts)
        # The prospective calibration files are expressly public; the banned
        # words below apply to private *paths*, not these reviewed filenames.
        if lowered.endswith((".sqlite", ".sqlite-wal", ".sqlite-shm", ".zip", ".pt", ".pth")):
            raise ValueError(f"Forbidden public file type: {rel}")
        if parts & FORBIDDEN_PATH_PARTS:
            raise ValueError(f"Forbidden private path: {rel}")
        payload = (ROOT / rel).read_bytes()
        for pattern in SECRET_PATTERNS:
            if pattern.search(payload):
                raise ValueError(f"Sensitive content pattern in {rel}")


def seal() -> None:
    scan()
    rows = [f"{sha(ROOT / rel)}  {rel}\n" for rel in sorted(FILES)]
    MANIFEST.write_bytes("".join(rows).encode("utf-8"))


def verify() -> None:
    scan()
    if not MANIFEST.is_file():
        raise ValueError("MANIFEST.sha256 missing")
    expected = [f"{sha(ROOT / rel)}  {rel}\n" for rel in sorted(FILES)]
    if MANIFEST.read_bytes() != "".join(expected).encode("utf-8"):
        raise ValueError("MANIFEST.sha256 mismatch")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"seal", "verify"}:
        raise SystemExit("usage: python tools/check_public.py {seal|verify}")
    (seal if sys.argv[1] == "seal" else verify)()
    print("PASS_PUBLIC_FILE_ALLOWLIST_AND_SHA256")
