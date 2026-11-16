import joblib
import pandas as pd
from src.config import MODELS_DIR
import src.features

def cargar_modelo():
    return joblib.load(MODELS_DIR / "modelo_riesgo_v1.pkl")

def predecir_riesgo(datos_atleta: dict) -> dict:
    modelo = cargar_modelo()
    X = pd.DataFrame([datos_atleta])
    proba = float(modelo.predict_proba(X)[0, 1])
    nivel = "bajo" if proba < 0.33 else "medio" if proba < 0.66 else "alto"
    return {
        "probabilidad_lesion": round(proba, 4),
        "clase_binaria": int(modelo.predict(X)[0]),
        "nivel_alerta": nivel,
    }

if __name__ == "__main__":
    ejemplo = {
        "posicion": "opuesto",
        "saltos_totales": 180, "remates": 60, "bloqueos": 25,
        "recepciones": 40, "desplazamientos_explosivos": 55,
        "minutos_entrenamiento": 120, "rpe": 9,
        "acwr": 1.5,
        "horas_sueno": 5.5, "calidad_sueno": 2,
        "dias_descanso_semana": 1, "horas_recuperacion_entre_sesiones": 8,
        "lesiones_previas": 2, "dolor_muscular_actual": 7,
        "edad": 21, "imc": 23.5,
        "movilidad_hombro": 160, "movilidad_tobillo": 45,
        "zona_lesion_previa": "hombro",
    }
    print(predecir_riesgo(ejemplo))