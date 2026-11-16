from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

TARGET = "lesion_semana_siguiente"
RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURES_NUMERICAS = [
    "saltos_totales", "remates", "bloqueos", "recepciones",
    "desplazamientos_explosivos", "minutos_entrenamiento", "rpe",
    "acwr",
    "horas_sueno", "calidad_sueno", "dias_descanso_semana",
    "horas_recuperacion_entre_sesiones",
    "lesiones_previas", "dolor_muscular_actual",
    "edad", "imc", "movilidad_hombro", "movilidad_tobillo",
]

FEATURES_CATEGORICAS = ["posicion", "zona_lesion_previa"]