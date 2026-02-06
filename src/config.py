# src/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    ROOT: Path = Path(__file__).resolve().parents[1]

    RAW_ZIP: Path = ROOT / "data" / "raw" / "heartdisease.zip"
    PROCESSED_CSV: Path = ROOT / "data" / "processed" / "heart_clean.csv"

    TARGET: str = "target"

    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    OUTPUT_DIR: Path = ROOT / "outputs"
    FIG_DPI: int = 140

CFG = Config()
