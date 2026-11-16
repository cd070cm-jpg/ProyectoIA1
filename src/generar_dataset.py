"""
Genera un dataset sintético realista de lesiones musculares en voleibolistas
de alto rendimiento (UPDS).
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N_ATLETAS = 20
SESIONES_POR_ATLETA = 120

POSICIONES = ["acomodador", "opuesto", "central", "receptor", "libero"]
ZONAS = ["hombro", "rodilla", "tobillo", "lumbar", "ninguna"]

PERFIL_POSICION = {
    "acomodador": (60, 15, 20, 0.8),
    "opuesto":    (140, 55, 30, 1.2),
    "central":    (130, 25, 60, 1.3),
    "receptor":   (110, 40, 25, 1.1),
    "libero":     (30, 5, 5, 0.7),
}

def simular_atleta(id_atleta: int) -> pd.DataFrame:
    posicion = RNG.choice(POSICIONES, p=[0.2, 0.2, 0.2, 0.3, 0.1])
    saltos_b, remates_b, bloqueos_b, factor_riesgo = PERFIL_POSICION[posicion]

    edad = int(RNG.integers(18, 26))
    imc = float(RNG.normal(23, 1.5))
    lesiones_previas = int(RNG.poisson(0.8))
    zona_previa = RNG.choice(ZONAS, p=[0.25, 0.25, 0.2, 0.15, 0.15]) if lesiones_previas > 0 else "ninguna"

    movilidad_hombro = float(np.clip(RNG.normal(165, 8), 130, 180))
    movilidad_tobillo = float(np.clip(RNG.normal(48, 6), 30, 60))

    filas = []
    carga_hist = []

    for dia in range(SESIONES_POR_ATLETA):
        dia_semana = dia % 7
        factor_partido = 1.4 if dia_semana == 5 else 1.0
        factor_descanso_semanal = 0.6 if dia_semana == 6 else 1.0

        saltos = max(0, int(RNG.normal(saltos_b * factor_partido * factor_descanso_semanal, 15)))
        remates = max(0, int(RNG.normal(remates_b * factor_partido * factor_descanso_semanal, 8)))
        bloqueos = max(0, int(RNG.normal(bloqueos_b * factor_partido * factor_descanso_semanal, 6)))
        recepciones = max(0, int(RNG.normal(35 * factor_partido * factor_descanso_semanal, 10)))
        desplazamientos = max(0, int(RNG.normal(50 * factor_partido * factor_descanso_semanal, 12)))

        minutos = float(np.clip(RNG.normal(100 * factor_partido * factor_descanso_semanal, 15), 40, 180))
        rpe = float(np.clip(RNG.normal(7 * factor_partido * factor_descanso_semanal, 1.2), 2, 10))

        carga_hist.append(rpe * minutos)

        if len(carga_hist) >= 7:
            carga_aguda = float(np.mean(carga_hist[-7:]))
        else:
            carga_aguda = float(np.mean(carga_hist))

        if len(carga_hist) >= 28:
            carga_cronica = float(np.mean(carga_hist[-28:]))
        else:
            carga_cronica = float(np.mean(carga_hist))

        acwr = round(carga_aguda / carga_cronica, 3) if carga_cronica > 0 else 1.0

        horas_sueno = float(np.clip(RNG.normal(7 - 0.15 * (rpe - 5), 0.9), 3.5, 10))
        calidad_sueno = int(np.clip(round(horas_sueno - 1 + RNG.normal(0, 0.8)), 1, 5))
        dias_descanso = int(np.clip(RNG.poisson(1.5), 0, 4))
        horas_recuperacion = float(np.clip(RNG.normal(18 - 1.5 * (rpe - 5), 4), 6, 36))

        dolor_base = (
            2.0
            + 2.5 * max(0, acwr - 1.1)
            + 0.6 * lesiones_previas
            - 0.4 * (calidad_sueno - 3)
            - 0.2 * dias_descanso
        )
        dolor = float(np.clip(RNG.normal(dolor_base, 1.2), 0, 10))

        riesgo_latente = (
            1.8 * max(0, acwr - 1.3)
            + 0.35 * (10 - horas_sueno)
            + 0.25 * dolor
            + 0.40 * lesiones_previas
            + 0.02 * max(0, 165 - movilidad_hombro)
            + 0.03 * max(0, 48 - movilidad_tobillo)
            - 0.15 * dias_descanso
            - 0.20 * (calidad_sueno - 3)
            + 0.5 * factor_riesgo
            + RNG.normal(0, 0.6)
        )

        prob_lesion = 1 / (1 + np.exp(-(riesgo_latente - 4.5)))
        lesion = int(RNG.random() < prob_lesion)

        filas.append({
            "id_atleta": f"ATL{id_atleta:03d}",
            "fecha": pd.Timestamp("2024-01-01") + pd.Timedelta(days=dia),
            "posicion": posicion,
            "edad": edad,
            "imc": round(imc, 2),
            "saltos_totales": saltos,
            "remates": remates,
            "bloqueos": bloqueos,
            "recepciones": recepciones,
            "desplazamientos_explosivos": desplazamientos,
            "minutos_entrenamiento": round(minutos, 1),
            "rpe": round(rpe, 1),
            "acwr": acwr,
            "horas_sueno": round(horas_sueno, 2),
            "calidad_sueno": calidad_sueno,
            "dias_descanso_semana": dias_descanso,
            "horas_recuperacion_entre_sesiones": round(horas_recuperacion, 1),
            "lesiones_previas": lesiones_previas,
            "zona_lesion_previa": zona_previa,
            "dolor_muscular_actual": round(dolor, 1),
            "movilidad_hombro": round(movilidad_hombro, 1),
            "movilidad_tobillo": round(movilidad_tobillo, 1),
            "lesion_semana_siguiente": lesion,
        })

    return pd.DataFrame(filas)

def main():
    todos = [simular_atleta(i + 1) for i in range(N_ATLETAS)]
    df = pd.concat(todos, ignore_index=True)

    salida = Path("data/raw/atletas_upds.csv")
    salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(salida, index=False)

    print(f"✅ Dataset generado: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"   Atletas: {df['id_atleta'].nunique()}")
    print(f"   Rango de fechas: {df['fecha'].min().date()} → {df['fecha'].max().date()}")
    print(f"\n   Distribución del target 'lesion_semana_siguiente':")
    print(df["lesion_semana_siguiente"].value_counts(normalize=True).round(3))
    print(f"\n   Guardado en: {salida.resolve()}")

if __name__ == "__main__":
    main()