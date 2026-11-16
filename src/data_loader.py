import pandas as pd
from src.config import DATA_RAW, DATA_PROCESSED, TARGET

def cargar_datos() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "atletas_upds.csv", parse_dates=["fecha"])

def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET])

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    # Incluye "str" para silenciar Pandas4Warning
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].fillna("desconocido")

    df = df[(df["horas_sueno"] >= 0) & (df["horas_sueno"] <= 14)]
    df = df[(df["rpe"] >= 1) & (df["rpe"] <= 10)]
    df = df[df["acwr"] < 3.0]

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED / "dataset_entrenamiento.csv", index=False)
    return df

if __name__ == "__main__":
    df = limpiar_datos(cargar_datos())
    print(f"Dataset limpio: {df.shape}")
    print(df[TARGET].value_counts(normalize=True).round(3))