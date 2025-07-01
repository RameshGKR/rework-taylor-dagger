"""get_expert_action.py (CLI version)

Compute expert (MPC) actions for each row of a pre-processed CSV from JuliaReach.

Usage
-----
    python get_expert_action.py  input.csv           # writes input_expert_filled.csv
    python get_expert_action.py  input.csv  output.csv

The script expects the CSV to contain at least:
    input_index, input_x1, input_y1, input_theta0, input_theta1
and will (re)write the columns `output_v0` and `output_delta0`.
"""

import sys
from pathlib import Path

import pandas as pd
from casadi import vertcat
from truck_trailer_multistage_loop_expert_function import get_expert_action_truck_trailer

def main() -> None:
    if len(sys.argv) not in (2, 3):
        sys.exit(
            """Usage:
    python get_expert_action.py  input.csv [output.csv]

If output.csv is omitted the result is written to
<basename>_expert_filled.csv in the same directory."""
        )

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) == 3 else input_path.with_name(
        input_path.stem + "_expert_filled.csv"
    )

    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # Prepare containers
    v0, delta0 = [], []

    for _, row in df.iterrows():
        # Skip rows with missing index
        if pd.isna(row["input_index"]):
            continue

        inp = vertcat(
            row["input_theta1"], row["input_x1"], row["input_y1"], row["input_theta0"]
        )
        idx = int(row["input_index"])
        out = get_expert_action_truck_trailer(inp, idx)
        v0.append(float(out[0]))
        delta0.append(float(out[1]))

    # Write/overwrite action columns
    df["output_v0"] = v0
    df["output_delta0"] = delta0

    df.to_csv(output_path, index=False)
    print(f"Expert actions written to {output_path}")


if __name__ == "__main__":
    main()
