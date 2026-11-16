"""
Dashboard interactivo - Sistema de Predicción de Lesiones Musculares (UPDS)
Diseño UI/UX mejorado: Formulario y Predicción en 2 columnas simultáneas + Base de Datos SQLite.
"""

import sys
from pathlib import Path
import sqlite3

# FIX: agregar la raíz del proyecto al sys.path para poder importar 'src'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import MODELS_DIR, REPORTS_DIR, DATA_PROCESSED, TARGET
import src.features

# =====================================================================
# CONFIGURACIÓN DE PÁGINA Y CSS PERSONALIZADO
# =====================================================================
st.set_page_config(
    page_title="UPDS Voleibol - Predicción de Lesiones",
    page_icon="🏐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ocultar menú de Streamlit y mejorar estilos de las tarjetas
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Estilo para las tarjetas de métricas */
    div[data-testid="metric-container"] {
        background-color: #1E2530;
        border: 1px solid #2E3B4E;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: #00B4D8;
    }
    
    /* Estilo para los tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E2530;
        border-bottom: 3px solid #00B4D8;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# BASE DE DATOS (SQLite)
# =====================================================================
DB_PATH = Path(__file__).resolve().parent.parent / "sistema_lesiones.db"

def init_db():
    """Crea la tabla si no existe al iniciar la app."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS historial_predicciones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                nombre_atleta TEXT NOT NULL,
                posicion TEXT,
                probabilidad REAL,
                nivel TEXT,
                acwr REAL,
                dolor REAL,
                horas_sueno REAL
            )
        """)

def guardar_prediccion(nombre, posicion, proba, nivel, acwr, dolor, horas_sueno):
    """Inserta el resultado en la base de datos."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO historial_predicciones 
            (nombre_atleta, posicion, probabilidad, nivel, acwr, dolor, horas_sueno)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (nombre, posicion, proba, nivel, acwr, dolor, horas_sueno))

def obtener_historial():
    """Recupera las últimas 50 predicciones."""
    with sqlite3.connect(DB_PATH) as conn:
        query = "SELECT fecha_registro, nombre_atleta, posicion, probabilidad, nivel, acwr, dolor FROM historial_predicciones ORDER BY fecha_registro DESC LIMIT 50"
        return pd.read_sql_query(query, conn)

# Inicializar DB
init_db()

# =====================================================================
# CARGA DE MODELO Y DATOS
# =====================================================================
def _verificar_archivos():
    faltantes = []
    if not (DATA_PROCESSED / "dataset_entrenamiento.csv").exists():
        faltantes.append("data/processed/dataset_entrenamiento.csv")
    if not (MODELS_DIR / "modelo_riesgo_v1.pkl").exists():
        faltantes.append("models/modelo_riesgo_v1.pkl")

    if faltantes:
        st.error(
            "⚠️ **Faltan archivos requeridos:**\n\n"
            + "\n".join(f"- `{f}`" for f in faltantes)
            + "\n\n**Ejecuta en la terminal:**\n"
            "```powershell\n"
            "python -m src.generar_dataset\n"
            "python -m src.data_loader\n"
            "python -m src.train\n"
            "```"
        )
        st.stop()

_verificar_archivos()

@st.cache_resource
def cargar_modelo():
    return joblib.load(MODELS_DIR / "modelo_riesgo_v1.pkl")

@st.cache_data
def cargar_dataset():
    return pd.read_csv(DATA_PROCESSED / "dataset_entrenamiento.csv", parse_dates=["fecha"])

@st.cache_data
def cargar_metricas():
    ruta = REPORTS_DIR / "metricas.json"
    if ruta.exists():
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

modelo = cargar_modelo()
df = cargar_dataset()
metricas = cargar_metricas()

# =====================================================================
# BARRA LATERAL
# =====================================================================
st.sidebar.markdown("<h2 style='text-align: center; color: #00B4D8;'>🏐 UPDS Voleibol</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
seccion = st.sidebar.radio(
    "**Módulos del Sistema**",
    [
        "🎯 Predicción Individual",
        "📊 Rendimiento del Modelo",
        "📈 Análisis de Datos",
        "🚨 Panel de Alertas",
        "🗂️ Historial de Atletas", # <-- NUEVA OPCIÓN AÑADIDA
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Ficha Técnica")
if metricas:
    st.sidebar.info(
        f"**Algoritmo:** {metricas['mejor_modelo'].capitalize()}\n\n"
        f"**ROC-AUC:** {metricas['resultados'][0]['roc_auc']:.3f}\n"
        f"**Sensibilidad:** {metricas['resultados'][0]['recall']:.3f}"
    )

st.sidebar.caption(f"Base de datos:\n{len(df):,} sesiones · {df['id_atleta'].nunique()} atletas")

# =====================================================================
# ENCABEZADO PRINCIPAL
# =====================================================================
st.title("Sistema de Predicción de Lesiones Musculares")
st.markdown("**Universidad Privada Domingo Savio** — *Selección de Voleibol de Alto Rendimiento*")
st.markdown("---")

# =====================================================================
# VISTA 1 — PREDICCIÓN INDIVIDUAL (2 COLUMNAS)
# =====================================================================
if seccion == "🎯 Predicción Individual":
    st.header("Evaluación de Riesgo de Lesión")
    
    col_formulario, col_resultado = st.columns([1.1, 1], gap="large")

    with col_formulario:
        st.markdown("### 📝 Datos del Atleta")
        st.write("Ingresa los datos para calcular la probabilidad de lesión en los próximos 7 días.")
        
        # NUEVO CAMPO: Nombre del atleta
        nombre_atleta = st.text_input("👤 Nombre completo del atleta", placeholder="Ej. Juan Pérez")
        
        with st.expander("🏐 Carga Física", expanded=True):
            c1, c2 = st.columns(2)
            posicion = c1.selectbox("Posición", ["acomodador", "opuesto", "central", "receptor", "libero"])
            minutos = c2.slider("Minutos entrenados", 30, 200, 100, step=10)
            
            saltos = c1.number_input("Saltos totales", 0, 300, 140, step=10)
            remates = c2.number_input("Remates", 0, 100, 45, step=5)
            bloqueos = c1.number_input("Bloqueos", 0, 100, 30, step=5)
            recepciones = c2.number_input("Recepciones", 0, 100, 35, step=5)
            desplazamientos = c1.number_input("Desplazamientos", 0, 150, 55, step=5)
            acwr = c2.slider("ACWR (Carga)", 0.5, 2.5, 1.2, step=0.05)
            rpe = st.slider("RPE (Esfuerzo percibido)", 1.0, 10.0, 7.0, step=0.5)

        with st.expander("😴 Descanso y Antecedentes", expanded=True):
            c1, c2 = st.columns(2)
            horas_sueno = c1.slider("Horas de sueño", 3.0, 12.0, 7.0, step=0.5)
            calidad_sueno = c2.slider("Calidad del sueño (1-5)", 1, 5, 3)
            dias_descanso = c1.slider("Días descanso/semana", 0, 5, 1)
            horas_recup = c2.slider("Horas entre sesiones", 6.0, 36.0, 18.0, step=0.5)
            lesiones_previas = c1.slider("Lesiones previas", 0, 5, 1)
            zona_previa = c2.selectbox("Zona de lesión previa", ["ninguna", "hombro", "rodilla", "tobillo", "lumbar"])

        with st.expander("🏥 Datos Clínicos", expanded=True):
            c1, c2 = st.columns(2)
            edad = c1.slider("Edad", 17, 30, 21)
            imc = c2.slider("IMC", 18.0, 30.0, 23.0, step=0.1)
            dolor = c1.slider("Dolor actual (0-10)", 0.0, 10.0, 3.0, step=0.5)
            mov_hombro = c2.slider("Mov. hombro (°)", 120, 180, 165)
            mov_tobillo = st.slider("Mov. tobillo (°)", 30, 60, 48)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔍 Calcular y Guardar Riesgo", type="primary", use_container_width=True):
            # Validar que se haya ingresado un nombre
            if not nombre_atleta.strip():
                st.warning("⚠️ Por favor, ingresa el nombre del atleta para poder guardar el registro.")
            else:
                datos = {
                    "posicion": posicion, "saltos_totales": saltos, "remates": remates,
                    "bloqueos": bloqueos, "recepciones": recepciones,
                    "desplazamientos_explosivos": desplazamientos,
                    "minutos_entrenamiento": minutos, "rpe": rpe, "acwr": acwr,
                    "horas_sueno": horas_sueno, "calidad_sueno": calidad_sueno,
                    "dias_descanso_semana": dias_descanso,
                    "horas_recuperacion_entre_sesiones": horas_recup,
                    "lesiones_previas": lesiones_previas, "dolor_muscular_actual": dolor,
                    "edad": edad, "imc": imc,
                    "movilidad_hombro": mov_hombro, "movilidad_tobillo": mov_tobillo,
                    "zona_lesion_previa": zona_previa,
                }
                X = pd.DataFrame([datos])
                proba = float(modelo.predict_proba(X)[0, 1])
                
                if proba < 0.33:
                    nivel, color, emoji = "BAJO", "#2ECC71", "🟢"
                elif proba < 0.66:
                    nivel, color, emoji = "MEDIO", "#F39C12", "🟡"
                else:
                    nivel, color, emoji = "ALTO", "#E74C3C", "🔴"
                    
                # GUARDAR EN BASE DE DATOS
                guardar_prediccion(nombre_atleta, posicion, proba, nivel, acwr, dolor, horas_sueno)
                    
                # Guardar en la memoria de la sesión
                st.session_state['prediccion'] = {
                    "proba": proba, "nivel": nivel, "color": color, "emoji": emoji, "nombre": nombre_atleta
                }

    with col_resultado:
        if 'prediccion' not in st.session_state:
            st.info("👈 Completa el formulario y presiona **Calcular** para ver los resultados aquí.")
        else:
            res = st.session_state['prediccion']
            proba = res["proba"]
            nivel = res["nivel"]
            color = res["color"]
            emoji = res["emoji"]
            nombre = res["nombre"]

            st.markdown(f"<h3 style='color: {color}; text-align: center;'>Paciente: {nombre}</h3>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='color: {color}; text-align: center;'>Riesgo {nivel} {emoji}</h4>", unsafe_allow_html=True)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"size": 60, "color": color}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#888"},
                    "bar": {"color": color, "thickness": 0.8},
                    "bgcolor": "#1E2530",
                    "steps": [
                        {"range": [0, 33], "color": "rgba(46, 204, 113, 0.15)"},
                        {"range": [33, 66], "color": "rgba(243, 156, 18, 0.15)"},
                        {"range": [66, 100], "color": "rgba(231, 76, 60, 0.15)"},
                    ],
                },
            ))
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

            # MENSAJE DE ÉXITO AL GUARDAR
            st.success("✅ Los datos han sido guardados en el historial exitosamente.")

            st.markdown("#### Recomendaciones")
            if nivel == "ALTO":
                st.error(
                    "- **Suspender entrenamiento** de alta intensidad por 48 h.\n"
                    "- **Evaluación clínica prioritaria** (hombros/rodillas/tobillo/zona lumbar).\n"
                    "- Revisar ACWR: si es > 1.5, hay sobrecarga aguda.\n"
                    "- Priorizar sueño (>7 h) y trabajo de movilidad."
                )
            elif nivel == "MEDIO":
                st.warning(
                    "- **Reducir volumen** de saltos y remates en 20-30%.\n"
                    "- Aumentar horas de sueño esta semana.\n"
                    "- Monitoreo diario de dolor muscular antes de entrenar."
                )
            else:
                st.success(
                    "- **Continuar con planificación** actual.\n"
                    "- Mantener rutina de descanso.\n"
                    "- Sin signos de sobrecarga inminente."
                )

# =====================================================================
# VISTA NUEVA — HISTORIAL DE ATLETAS
# =====================================================================
elif seccion == "🗂️ Historial de Atletas":
    st.header("Historial de Predicciones Guardadas")
    st.write("Registro histórico de los atletas evaluados en este sistema (últimas 50 evaluaciones).")
    
    try:
        df_hist = obtener_historial()
        if df_hist.empty:
            st.info("Aún no hay registros guardados. Ve a 'Predicción Individual' y guarda el primero.")
        else:
            df_hist.columns = ["Fecha de Registro", "Nombre del Atleta", "Posición", "Probabilidad", "Nivel de Riesgo", "ACWR", "Dolor"]
            df_hist["Probabilidad"] = (df_hist["Probabilidad"] * 100).round(1)
            df_hist["Fecha de Registro"] = pd.to_datetime(df_hist["Fecha de Registro"]).dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(
                df_hist, use_container_width=True, hide_index=True,
                column_config={
                    "Probabilidad": st.column_config.ProgressColumn("Prob. (%)", min_value=0, max_value=100, format="%.1f%%"),
                }
            )
    except Exception as e:
        st.error(f"Error al leer la base de datos: {e}")

# =====================================================================
# VISTA 2 — MÉTRICAS DEL MODELO
# =====================================================================
elif seccion == "📊 Rendimiento del Modelo":
    st.header("Evaluación del Algoritmo Predictivo")

    if not metricas:
        st.warning("Ejecuta `python -m src.train` para generar las métricas.")
        st.stop()

    df_m = pd.DataFrame(metricas["resultados"])
    mejor = metricas["mejor_modelo"]

    cols = st.columns(4)
    mejor_data = df_m[df_m["modelo"] == mejor].iloc[0]
    cols[0].metric("Modelo Seleccionado", mejor.capitalize())
    cols[1].metric("ROC-AUC", f"{mejor_data['roc_auc']:.4f}")
    cols[2].metric("Sensibilidad (Recall)", f"{mejor_data['recall']:.4f}")
    cols[3].metric("Precisión", f"{mejor_data['precision']:.4f}")

    st.markdown("---")
    
    col_tabla, col_grafico = st.columns([1, 2], gap="large")
    
    with col_tabla:
        st.subheader("Comparativa de Algoritmos")
        st.dataframe(
            df_m.rename(columns={"modelo": "Modelo", "roc_auc": "ROC-AUC", "precision": "Precisión", "recall": "Recall", "f1": "F1-Score"}),
            use_container_width=True, hide_index=True
        )
        st.info("El modelo fue seleccionado maximizando el ROC-AUC. Se prioriza el Recall para minimizar los falsos negativos (atletas en riesgo no detectados).")

    with col_grafico:
        st.subheader("Gráfico Comparativo")
        df_long = df_m.melt(
            id_vars="modelo",
            value_vars=["roc_auc", "precision", "recall", "f1"],
            var_name="Métrica", value_name="Valor",
        )
        fig = px.bar(
            df_long, x="modelo", y="Valor", color="Métrica",
            barmode="group", text_auto=".3f",
            color_discrete_sequence=["#00B4D8", "#F39C12", "#2ECC71", "#E74C3C"],
        )
        fig.update_layout(
            height=400, yaxis_range=[0, 1],
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# VISTA 3 — ANÁLISIS DEL DATASET
# =====================================================================
elif seccion == "📈 Análisis de Datos":
    st.header("Análisis Exploratorio del Dataset")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sesiones Registradas", f"{len(df):,}")
    c2.metric("Atletas Evaluados", df["id_atleta"].nunique())
    c3.metric("Tasa de Lesión Histórica", f"{df[TARGET].mean()*100:.1f}%")
    c4.metric("Variables Analizadas", df.shape[1] - 3)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Distribución", "🔥 Correlaciones", "📉 Riesgo (ACWR)", "🏐 Por Posición"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                df, names=TARGET, hole=0.5,
                color=TARGET, color_discrete_map={0: "#2ECC71", 1: "#E74C3C"},
                labels={TARGET: "Lesión próxima semana"},
                title="Balance del Target (0 = Sano, 1 = Lesión)"
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        cols_num = df.select_dtypes(include="number").columns.tolist()
        corr = df[cols_num].corr()
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Matriz de Correlación de Pearson"
        )
        fig.update_layout(height=600, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = px.scatter(
            df.sample(min(1000, len(df)), random_state=42),
            x="acwr", y="dolor_muscular_actual",
            color=TARGET, color_discrete_map={0: "#2ECC71", 1: "#E74C3C"},
            opacity=0.6, labels={TARGET: "Lesión", "acwr": "ACWR", "dolor_muscular_actual": "Dolor (0-10)"},
            title="Relación entre Carga Aguda/Crónica (ACWR) y Dolor Muscular"
        )
        fig.add_vline(x=1.5, line_dash="dash", line_color="#F39C12",
                      annotation_text="Umbral de Riesgo (1.5)")
        fig.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        tasa = df.groupby("posicion")[TARGET].agg(["mean", "count"]).reset_index()
        tasa.columns = ["posicion", "tasa_lesion", "n_sesiones"]
        fig = px.bar(
            tasa, x="posicion", y="tasa_lesion", text_auto=".1%",
            color="tasa_lesion", color_continuous_scale="Reds",
            title="Tasa de Lesión por Posición en Cancha"
        )
        fig.update_layout(height=450, yaxis_tickformat=".0%", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# VISTA 4 — ALERTAS POR ATLETA
# =====================================================================
elif seccion == "🚨 Panel de Alertas":
    st.header("Panel de Control del Cuerpo Técnico")
    st.write("Evaluación del riesgo inminente basada en la última sesión registrada de cada atleta.")

    feats = df.drop(columns=[TARGET, "id_atleta", "fecha"], errors="ignore")
    df_pred = df.copy()
    df_pred["prob_lesion"] = modelo.predict_proba(feats)[:, 1]
    df_pred["nivel"] = pd.cut(
        df_pred["prob_lesion"],
        bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["🟢 Bajo", "🟡 Medio", "🔴 Alto"],
    )

    ultimas = (
        df_pred.sort_values("fecha")
        .groupby("id_atleta")
        .tail(1)
        .sort_values("prob_lesion", ascending=False)
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Atletas Monitoreados", len(ultimas))
    c2.metric("🔴 Alertas Altas", (ultimas["nivel"] == "🔴 Alto").sum())
    c3.metric("🟡 Alertas Medias", (ultimas["nivel"] == "🟡 Medio").sum())
    c4.metric("🟢 Sin Alerta", (ultimas["nivel"] == "🟢 Bajo").sum())

    st.markdown("---")
    
    col_tabla, col_grafico = st.columns([1, 1], gap="large")

    with col_tabla:
        st.subheader("Ranking de Riesgo Actual")
        tabla = ultimas[["id_atleta", "posicion", "fecha", "prob_lesion", "nivel"]].copy()
        tabla["prob_lesion"] = (tabla["prob_lesion"] * 100).round(1)
        tabla.columns = ["Atleta", "Posición", "Última Sesión", "Prob. (%)", "Nivel"]
        tabla["Última Sesión"] = tabla["Última Sesión"].dt.date

        st.dataframe(
            tabla, use_container_width=True, hide_index=True,
            column_config={
                "Prob. (%)": st.column_config.ProgressColumn(
                    "Prob. (%)", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
        )

    with col_grafico:
        st.subheader("Distribución de Riesgo")
        fig = px.bar(
            tabla.sort_values("Prob. (%)"),
            x="Prob. (%)", y="Atleta", orientation="h",
            color="Nivel", text="Prob. (%)",
            color_discrete_map={"🟢 Bajo": "#2ECC71", "🟡 Medio": "#F39C12", "🔴 Alto": "#E74C3C"},
        )
        fig.update_layout(height=500, xaxis_range=[0, 100], paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Evolución Histórica del Atleta")
    atleta = st.selectbox("Selecciona un atleta para ver su historial de riesgo:", sorted(df_pred["id_atleta"].unique()))
    
    hist = df_pred[df_pred["id_atleta"] == atleta].sort_values("fecha")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["fecha"], y=hist["prob_lesion"] * 100,
        mode="lines+markers", name="Probabilidad",
        line=dict(color="#00B4D8", width=3),
        marker=dict(size=6, color="#FAFAFA", line=dict(width=2, color="#00B4D8"))
    ))
    fig.add_hline(y=33, line_dash="dot", line_color="#F39C12", annotation_text="Riesgo Medio")
    fig.add_hline(y=66, line_dash="dot", line_color="#E74C3C", annotation_text="Riesgo Alto")
    
    fig.update_layout(
        height=400, xaxis_title="Fecha de Entrenamiento", yaxis_title="Probabilidad de Lesión (%)",
        yaxis_range=[0, 100], paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)