#!/usr/bin/env STAIR-env
import os
import subprocess
import sys
from pathlib import Path


def _python_has_scanpy(python_exe: str) -> bool:
    try:
        subprocess.run(
            [python_exe, "-c", "import scanpy"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def _candidate_pythons():
    configured_python = os.environ.get("HYPERMOA_PYTHON")
    if configured_python:
        yield Path(configured_python)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        yield Path(conda_prefix) / "bin" / "python"

    current_python = Path(sys.executable).resolve()
    try:
        conda_root = current_python.parents[1]
        envs_dir = conda_root / "envs"
        if envs_dir.exists():
            yield from sorted(envs_dir.glob("*/bin/python"))
    except IndexError:
        pass

    default_envs_dir = Path(sys.prefix).parent / "envs"
    if default_envs_dir.exists():
        yield from sorted(default_envs_dir.glob("*/bin/python"))


def resolve_python() -> str:
    if _python_has_scanpy(sys.executable):
        return sys.executable

    seen = set()
    for python_path in _candidate_pythons():
        python_path = python_path.expanduser().resolve()
        if python_path in seen or not python_path.exists():
            continue
        seen.add(python_path)
        if _python_has_scanpy(str(python_path)):
            print(f"[INFO] Current Python lacks scanpy; using {python_path}")
            return str(python_path)

    return sys.executable


def run_step(script_path: Path, python_exe: str) -> None:
    print(f"[RUN] {script_path}")
    subprocess.run([python_exe, str(script_path)], check=True)


def configure_runtime_threads(python_exe: str) -> None:
    target_threads = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    for key, target_value in target_threads.items():
        current = os.environ.get(key)
        if current != target_value:
            os.environ[key] = target_value

    env_prefix = str(Path(python_exe).resolve().parent.parent)
    os.environ["CONDA_PREFIX"] = env_prefix
    os.environ["PATH"] = str(Path(env_prefix) / "bin") + os.pathsep + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = str(Path(env_prefix) / "lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    if (Path(env_prefix) / "lib" / "R").exists():
        os.environ["R_HOME"] = str(Path(env_prefix) / "lib" / "R")


def main() -> int:
    python_exe = resolve_python()
    configure_runtime_threads(python_exe)

    repo_root = Path(__file__).resolve().parent.parent
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [str(repo_root)]
    if current_pythonpath:
        pythonpath_parts.append(current_pythonpath)
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    run_dir = repo_root / "Human_lymph_node_run"

    steps = [
        run_dir / "01_prepare_data.py",
        run_dir / "02_embedding_alignment.py",
        run_dir / "03_slice_order_and_z_reconstruction.py",
        run_dir / "04_location_alignment.py",
        run_dir / "05_build_3d_and_export.py",
    ]

    for step in steps:
        if not step.exists():
            raise FileNotFoundError(f"Missing pipeline step: {step}")

    try:
        for step in steps:
            run_step(step, python_exe)
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Pipeline failed at: {exc.cmd}")
        return exc.returncode

    print("Human lymph node pipeline finished. Outputs are under ./Human_lymph_node_result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
