import zipfile
import pandas as pd
from pathlib import Path

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

FILES = [
    "processed.cleveland.data",
    "processed.hungarian.data",
    "processed.switzerland.data",
    "processed.va.data"
]

def load_heart_disease_from_zip(zip_path: Path) -> pd.DataFrame:
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip introuvable : {zip_path}")

    dataframes = []

    with zipfile.ZipFile(zip_path, "r") as z:
        for fname in FILES:
            if fname not in z.namelist():
                raise FileNotFoundError(f"{fname} manquant dans le zip")

            with z.open(fname) as f:
                df = pd.read_csv(f, header=None, names=COLUMNS)
                dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)
