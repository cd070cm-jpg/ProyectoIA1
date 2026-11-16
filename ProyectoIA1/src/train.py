import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score
)

from src.config import (
    DATA_PROCESSED, MODELS_DIR, REPORTS_DIR,
    TARGET, RANDOM_STATE, TEST_SIZE
)
from src.features import construir_preprocesador

def cargar_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / "dataset_entrenamiento.csv")

def construir_modelos():
    pre = construir_preprocesador()
    return {
        "logistic": Pipeline([("pre", pre), ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))]),
        "random_forest": Pipeline([("pre", pre), ("clf", RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1))]),
        "xgboost": Pipeline([("pre", pre), ("clf", XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            scale_pos_weight=3, eval_metric="logloss",
            random_state=RANDOM_STATE))]),
    }

def evaluar(nombre, modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    m = {
        "modelo": nombre,
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }
    print(f"\n=== {nombre} ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
    return m

def main():
    df = cargar_dataset()
    X = df.drop(columns=[TARGET, "id_atleta", "fecha"], errors="ignore")
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    modelos = construir_modelos()
    resultados = []
    mejor_nombre, mejor_modelo, mejor_auc = None, None, -1

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        m = evaluar(nombre, modelo, X_test, y_test)
        resultados.append(m)
        if m["roc_auc"] > mejor_auc:
            mejor_auc, mejor_nombre, mejor_modelo = m["roc_auc"], nombre, modelo

    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    REPORTS_DIR.mkdir(exist_ok=True, parents=True)
    joblib.dump(mejor_modelo, MODELS_DIR / "modelo_riesgo_v1.pkl")

    with open(REPORTS_DIR / "metricas.json", "w", encoding="utf-8") as f:
        json.dump({"mejor_modelo": mejor_nombre, "resultados": resultados},
                  f, indent=2, ensure_ascii=False)

    print(f"\n✅ Mejor modelo: {mejor_nombre} (ROC-AUC = {mejor_auc})")
    print(f"   Guardado en {MODELS_DIR / 'modelo_riesgo_v1.pkl'}")

    print(f"\n📊 Validación GroupKFold por atleta ({mejor_nombre}):")
    mejor_pipe = construir_modelos()[mejor_nombre]
    gkf = GroupKFold(n_splits=5)
    aucs = cross_val_score(
        mejor_pipe, X, y,
        cv=gkf, groups=df["id_atleta"],
        scoring="roc_auc", n_jobs=-1
    )
    print(f"   AUC por fold: {[round(float(a), 3) for a in aucs]}")
    print(f"   AUC media   : {aucs.mean():.4f} ± {aucs.std():.4f}")

if __name__ == "__main__":
    main()