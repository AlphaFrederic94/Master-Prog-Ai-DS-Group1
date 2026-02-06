import pandas as pd

def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 40, 55, 70, 120],
        labels=["young", "middle", "senior", "elder"]
    )
    return df

def add_bmi_proxy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # proxy BMI (feature synthétique)
    df["weight_est"] = df["chol"] / 2.5
    df["bmi_est"] = df["weight_est"] / (1.70 ** 2)
    return df

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_chol_interaction"] = df["age"] * df["chol"]
    return df
