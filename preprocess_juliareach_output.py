"""Preprocess output CSV from JuliaReach before feeding to get_expert_action.py

1. Blank `output_v0` & `output_delta0`
2. Remove `time`
3. Re-index aggregation rows (0 1 1 1 1  1 2 2 2 2 …)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Drop separator lines coming from JuliaReach
    df = df.dropna(how="all").reset_index(drop=True)

    # 1. clear expert-action columns
    for col in ("output_v0", "output_delta0"):
        if col in df.columns:
            df[col] = np.nan

    # 2. remove time column
    if "time" in df.columns:
        df = df.drop(columns=["time"])

    # 3. fix the input_index pattern
    new_idx = df["input_index"].copy()
    i, n = 0, len(df)
    while i < n:
        orig = df.loc[i, "input_index"]
        j = i + 1
        while j < n and df.loc[j, "input_index"] == orig:
            j += 1
        if not pd.isna(orig):
            new_idx.iloc[i] = orig
            new_idx.iloc[i + 1 : j] = orig + 1
        i = j

    # final clean-up
    df["input_index"] = new_idx
    df = df[df["input_index"].notna()].copy()
    df["input_index"] = df["input_index"].astype(int)
    return df


def main() -> None:
    if len(sys.argv) not in (2, 3):
        sys.exit(
            "Usage: python preprocess_reachability_output.py  input.csv  [output.csv]"
        )

    inp = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) == 3 else inp.with_name(
        inp.stem + "_preprocessed.csv"
    )

    df = pd.read_csv(inp)
    df = preprocess(df)
    df.to_csv(out, index=False)
    print(f"Preprocessed CSV written to {out}")


if __name__ == "__main__":
    main()
