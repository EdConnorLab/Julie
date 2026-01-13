from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator
import pandas as pd


@dataclass(frozen=True)
class Session:
    date: str        # "YYYY-MM-DD"
    round_no: int
    location: str


def _normalize_date_to_str(x) -> str:
    if hasattr(x, "strftime"):
        return str(x.strftime("%Y-%m-%d"))
    return str(x)


def sessions_from_metadata(metadata_df: pd.DataFrame) -> Iterator[Session]:
    """
    Convert metadata rows into clean Session objects.
    Expects columns: 'Date', 'Round No.', 'Location'.
    """
    cols = ["Date", "Round No.", "Location"]
    missing = set(cols) - set(metadata_df.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")

    df = metadata_df[cols].drop_duplicates()

    # itertuples renames "Round No." -> Round_No_

    for r in df.itertuples(index=False):
        yield Session(
            date=_normalize_date_to_str(r[0]),  # Date
            round_no=int(r[1]),  # Round No.
            location=str(r[2]),  # Location
        )

@dataclass(frozen=True)
class SessionConfig:
    group_name: str
    use_sorted: bool = False
    bin_size: float = 0.05
    only_valid_channels: bool = False


class StatisticalTestConfig:
    alpha: float = 0.05
    n_permutations: int | None = 1000
    random_state: int | None = None

