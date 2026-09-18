"""What produced a number, recorded beside the number.

Every figure this project reports is regenerable only if the thing that
produced it can be identified later. "nDCG@5 was 0.9984" is not a reproducible
claim; "nDCG@5 was 0.9984 at commit e45ed19, on bge-small-en-v1.5, against
stress_a.jsonl with digest 3f9c…" is.

So every harness stamps its output with this block. The cost is a few
milliseconds and a dozen lines of JSON; the benefit is that a number in the
paper can be traced to the code and the data that made it, including by someone
who is not us and does not have this machine.

Recorded: the commit and whether the tree was dirty when the run happened, the
embedding model, the LLM chain that would have been used, digests of the
evaluation data files, and the versions of the libraries whose behaviour can
move a score.
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "evals" / "data"

#: Libraries whose version can change a score rather than merely how it is
#: printed. A pandas bump cannot move nDCG; an onnxruntime bump can.
TRACKED_PACKAGES = ("fastembed", "onnxruntime", "numpy", "pdfplumber", "pypdf", "tokenizers")


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10, check=False
        ).stdout.strip()
    except Exception:
        return ""


def _commit() -> dict:
    commit = _git("rev-parse", "HEAD")
    # A dirty tree means the commit does not describe the code that ran, which
    # is worth knowing before a number goes in a paper.
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    return {
        "commit": commit or "unknown",
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "dirty": dirty,
    }


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def dataset_digests() -> dict[str, str]:
    """Digest per evaluation data file, so a relabelled case is visible."""
    if not DATA_DIR.exists():
        return {}
    return {path.name: _digest(path) for path in sorted(DATA_DIR.glob("*.jsonl"))}


def _versions() -> dict[str, str]:
    found = {}
    for name in TRACKED_PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return found


def _models() -> dict:
    models: dict = {}
    try:
        from services.similarity import EMBEDDING_MODEL

        models["embedding"] = EMBEDDING_MODEL
    except Exception:
        pass
    try:
        from core.config import settings

        # The chain, not the model that answered: which rung a given call
        # landed on is a runtime fact, recorded per call by the caller.
        models["llm_chain"] = list(settings.model_chain)
        models["structured_output_mode"] = settings.structured_output_mode
    except Exception:
        pass
    return models


def provenance() -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": _commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _versions(),
        "models": _models(),
        "datasets": dataset_digests(),
    }


def render(block: dict) -> list[str]:
    git = block.get("git", {})
    lines = [
        "",
        "  PROVENANCE",
        f"    commit      {git.get('commit', '?')[:12]}"
        f"{'  (DIRTY TREE)' if git.get('dirty') else ''}",
        f"    generated   {block.get('generated_at', '?')}",
    ]
    models = block.get("models", {})
    if models.get("embedding"):
        lines.append(f"    embedding   {models['embedding']}")
    packages = block.get("packages", {})
    if packages:
        lines.append(
            "    packages    "
            + ", ".join(f"{name} {version}" for name, version in sorted(packages.items()))
        )
    datasets = block.get("datasets", {})
    if datasets:
        lines.append(
            "    data        "
            + ", ".join(f"{name}:{digest[:8]}" for name, digest in sorted(datasets.items()))
        )
    return lines
