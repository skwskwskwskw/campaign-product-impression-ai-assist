"""I/O helpers for metric pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def load_dataframe(path: str) -> pd.DataFrame:
    """Load a DataFrame from a CSV or Parquet path."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path_obj.suffix.lower() == ".parquet":
        return pd.read_parquet(path_obj)
    if path_obj.suffix.lower() == ".csv":
        return pd.read_csv(path_obj)

    raise ValueError("Unsupported file type. Use .csv or .parquet")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)


def write_outputs(
    outputs: Dict[str, pd.DataFrame],
    output_dir: str,
    formats: Iterable[str] = ("csv",),
) -> Dict[str, Dict[str, str]]:
    """
    Write output DataFrames to disk.

    Returns:
        Dict mapping output name to dict of {format: filepath}.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, str]] = {}
    for name, df in outputs.items():
        results[name] = {}
        for fmt in formats:
            fmt_lower = fmt.lower()
            out_path = output_path / f"{name}.{fmt_lower}"
            if fmt_lower == "csv":
                _write_csv(df, out_path)
            elif fmt_lower == "parquet":
                try:
                    _write_parquet(df, out_path)
                except Exception as exc:  # pragma: no cover - optional dependency
                    raise RuntimeError(
                        "Failed to write parquet. Ensure pyarrow or fastparquet is installed."
                    ) from exc
            else:
                raise ValueError(f"Unsupported format: {fmt}")

            results[name][fmt_lower] = str(out_path)

    return results
