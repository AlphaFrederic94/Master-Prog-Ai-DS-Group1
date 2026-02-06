import numpy as np
import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # valeurs manquantes codées par '?'
    df.replace("?", np.nan, inplace=True)

    # conversion numérique
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # imputation médiane
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    return df

def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 0 = absence, 1 = présence
    df["target"] = (df["target"] > 0).astype(int)
    return df
