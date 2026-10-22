import os
from pathlib import Path


def resolve_data_root(repo_root=None) -> Path:
    """Return the shared data directory for run scripts.

    Priority:
    1. HYPERMOA_DATA_DIR environment variable.
    2. data/datasets/ next to the repository parent.
    3. legacy data/ next to or inside the repository.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]
    else:
        repo_root = Path(repo_root).resolve()

    env_data_dir = os.environ.get("HYPERMOA_DATA_DIR")
    if env_data_dir:
        data_root = Path(env_data_dir).expanduser().resolve()
        if not data_root.exists():
            raise FileNotFoundError(f"Configured data directory does not exist: {data_root}")
        return data_root / "datasets" if (data_root / "datasets").is_dir() else data_root

    candidates = [
        repo_root.parent / "data" / "datasets",
        repo_root.parent / "data",
        repo_root / "data",
    ]
    for data_root in candidates:
        if data_root.exists():
            return data_root

    raise FileNotFoundError(
        "Data directory not found. Expected ../data/datasets, ../data or ./data relative to "
        f"repository root: {repo_root}"
    )
