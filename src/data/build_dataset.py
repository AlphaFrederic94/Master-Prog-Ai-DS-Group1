from pathlib import Path
from src.data.loader import load_heart_disease_from_zip
from src.data.preprocessing import clean_data, binarize_target
from src.features.engineering import (
    add_age_group, add_bmi_proxy, add_interaction_features
)

def build_dataset(zip_path: Path, out_path: Path):
    df = load_heart_disease_from_zip(zip_path)
    df = clean_data(df)
    df = binarize_target(df)

    df = add_age_group(df)
    df = add_bmi_proxy(df)
    df = add_interaction_features(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("Dataset final sauvegardé :", out_path)

if __name__ == "__main__":
    build_dataset(
        zip_path=Path("data/raw/heartdisease.zip"),
        out_path=Path("data/processed/heart_clean.csv")
    )
