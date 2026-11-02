from dataclasses import dataclass
from typing import Iterator, List, Tuple

import pandas as pd


@dataclass
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp    # inclusive
    test_start: pd.Timestamp
    test_end: pd.Timestamp     # inclusive


def build_trading_calendar(prepared_dfs: List[pd.DataFrame]) -> pd.Series:
    all_dates = pd.concat([df["Date"] for df in prepared_dfs])
    return pd.Series(sorted(all_dates.unique()))


def generate_walk_forward_folds(
    calendar: pd.Series,
    train_days: int = 500,
    test_days: int = 63,
    step_days: int = 63,
    embargo_days: int = 5,
) -> Iterator[Fold]:
    n = len(calendar)
    fold_id = 0
    train_start_idx = 0

    while True:
        train_end_idx = train_start_idx + train_days - 1
        test_start_idx = train_end_idx + embargo_days + 1
        test_end_idx = test_start_idx + test_days - 1

        if test_end_idx >= n:
            break

        yield Fold(
            fold_id=fold_id,
            train_start=calendar.iloc[train_start_idx],
            train_end=calendar.iloc[train_end_idx],
            test_start=calendar.iloc[test_start_idx],
            test_end=calendar.iloc[test_end_idx],
        )

        fold_id += 1
        train_start_idx += step_days


def slice_fold(df: pd.DataFrame, fold: Fold) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[(df["Date"] >= fold.train_start) & (df["Date"] <= fold.train_end)].copy()
    test_df = df[(df["Date"] >= fold.test_start) & (df["Date"] <= fold.test_end)].copy()
    return train_df, test_df
