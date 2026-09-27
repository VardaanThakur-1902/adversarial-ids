from pathlib import Path
import pandas as pd


def load_csv_files(data_dir):
    data_dir = Path(data_dir)

    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}"
        )

    dataframes = []

    for file in csv_files:
        print(f"Loading: {file.name}")

        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()

        dataframes.append(df)

    combined = pd.concat(
        dataframes,
        ignore_index=True
    )

    return combined