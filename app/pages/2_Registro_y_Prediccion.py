import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import plotly.express as px
import plotly.graph_objects as go

# Modelos solicitados
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DATA_PROCESSED, TARGET, RANDOM_STATE, FEATURES_NUMERICAS
from src.features import construir_preprocesador

st.set_page_config(page_title="Registro y Predicción", page_icon="📝", layout="wide")
st.title("📝 Registro de Sesión y Predicción")

DB_PATH = Path(__file__).resolve().parent.parent.parent / "sistema_lesiones.db"

# --- Cargar atletas ---
try:
    with sqlite3.connect(DB_PATH) as conn:
        df_atletas = pd.read_sql_query("SELECT nombre, posicion, edad, estatura, peso FROM atletas ORDER BY nombre", conn)
    lista_atletas = df_atletas["nombre"].tolist()
except:
    lista_atletas = []

if not lista_atletas:
    st.warning("⚠️ Primero debes registrar atletas en la pestaña 'Atletas'.")
    st.stop()

# --- Entrenar los 4 modelos ---
@st.cache_resource(show_spinner="Entrenando los 4 modelos de IA...")
def entrenar_modelos():
    df = pd.read_csv(DATA_PROCESSED / "dataset_entrenamiento.csv")
    X = df.drop(columns=[TARGET, "id_atleta", "fecha"], errors="ignore")
    y = df[TARGET]
    
    pre = construir_preprocesador()
    
    log_reg = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))]).fit(X, y)
    lin_reg = Pipeline([("pre", pre), ("reg", LinearRegression())]).fit(X, y)
    rf = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))]).fit(X, y)
    nn = Pipeline([("pre", pre), ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE))]).fit(X, y)
    
    return log_reg, lin_reg, rf, nn, df

log_reg, lin_reg, rf, nn, df_historico = entrenar_modelos()

# =====================================================================
# DIVISIÓN EN 2 COLUMNAS (FORMULARIO IZQ, RESULTADOS DER)
# =====================================================================
col_izq, col_der = st.columns([1, 1.3], gap="large")

# -------------------------------------------------------------
# COLUMNA IZQUIERDA: FORMULARIO
# -------------------------------------------------------------
with col_izq:
    st.markdown("### 📋 Datos de la Sesión")
    
    atleta_sel = st.selectbox("Selecciona el Atleta", lista_atletas)
    datos_atleta = df_atletas[df_atletas["nombre"] == atleta_sel].iloc[0]
    imc = datos_atleta["peso"] / (datos_atleta["estatura"]/100)**2

    with st.form("form_sesion"):
        acwr = st.slider("ACWR (Carga Aguda/Crónica)", 0.5, 2.5, 1.2, step=0.05)
        rpe = st.slider("RPE (Esfuerzo 1-10)", 1.0, 10.0, 7.0)
        minutos = st.slider("Minutos entrenados", 30, 200, 100)
        
        dolor = st.slider("Dolor muscular (0-10)", 0.0, 10.0, 3.0)
        horas_sueno = st.slider("Horas de sueño", 3.0, 12.0, 7.0)
        calidad_sueno = st.slider("Calidad del sueño (1-5)", 1, 5, 3)
        
        with st.expander("Variables complementarias", expanded=False):
            dias_descanso = st.slider("Días descanso/semana", 0, 5, 1)
            lesiones_previas = st.number_input("Lesiones previas", 0, 5, 0)
            zona_previa = st.selectbox("Zona lesión previa", ["ninguna", "hombro", "rodilla", "tobillo", "lumbar"])
            saltos = st.number_input("Saltos totales", 0, 300, 100)
            remates = st.number_input("Remates", 0, 100, 30)
            bloqueos = st.number_input("Bloqueos", 0, 100, 20)
            recepciones = st.number_input("Recepciones", 0, 100, 30)
            desplazamientos = st.number_input("Desplazamientos", 0, 150, 40)
            horas_recup = st.slider("Horas entre sesiones", 6.0, 36.0, 18.0)
            mov_hombro = st.slider("Mov. hombro (°)", 120, 180, 165)
            mov_tobillo = st.slider("Mov. tobillo (°)", 30, 60, 48)

        calcular = st.form_submit_button("🧠 Predecir con los 4 Modelos", type="primary", use_container_width=True)

    # Lógica de predicción y guardado
    if calcular:
        datos_entrada = pd.DataFrame([{
            "posicion": datos_atleta["posicion"], "edad": datos_atleta["edad"], "imc": round(imc, 1),
            "saltos_totales": saltos, "remates": remates, "bloqueos": bloqueos,
            "recepciones": recepciones, "desplazamientos_explosivos": desplazamientos,
            "minutos_entrenamiento": minutos, "rpe": rpe, "acwr": acwr,
            "horas_sueno": horas_sueno, "calidad_sueno": calidad_sueno,
            "dias_descanso_semana": dias_descanso, "horas_recuperacion_entre_sesiones": horas_recup,
            "lesiones_previas": lesiones_previas, "dolor_muscular_actual": dolor,
            "movilidad_hombro": mov_hombro, "movilidad_tobillo": mov_tobillo,
            "zona_lesion_previa": zona_previa
        }])

        try:
            prob_log = float(log_reg.predict_proba(datos_entrada)[0, 1])
            prob_lin = float(lin_reg.predict(datos_entrada)[0])
            prob_lin = max(0.0, min(1.0, prob_lin))
            prob_rf = float(rf.predict_proba(datos_entrada)[0, 1])
            prob_nn = float(nn.predict_proba(datos_entrada)[0, 1])
        except Exception as e:
            st.error(f"Error al predecir: {e}")
            st.stop()

        # Guardar en base de datos
        proba_final = prob_rf
        nivel_final = "ALTO" if proba_final > 0.66 else ("MEDIO" if proba_final > 0.33 else "BAJO")
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO historial_predicciones 
                (nombre_atleta, posicion, probabilidad, nivel, acwr, dolor, horas_sueno)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (atleta_sel, datos_atleta["posicion"], proba_final, nivel_final, acwr, dolor, horas_sueno))

        # GUARDAR EN MEMORIA PARA MOSTRAR EN LA DERECHA
        st.session_state['prediccion_4modelos'] = {
            "atleta": atleta_sel,
            "prob_log": prob_log, "prob_lin": prob_lin, "prob_rf": prob_rf, "prob_nn": prob_nn,
            "acwr": acwr, "dolor": dolor, "horas_sueno": horas_sueno, "minutos": minutos,
            "rpe": rpe, "lesiones_previas": lesiones_previas, "calidad_sueno": calidad_sueno,
            "nivel_final": nivel_final
        }

# -------------------------------------------------------------
# COLUMNA DERECHA: RESULTADOS Y GRÁFICOS (PESTAÑAS)
# -------------------------------------------------------------
with col_der:
    if 'prediccion_4modelos' not in st.session_state:
        st.info("👈 Completa el formulario a la izquierda y presiona **Predecir** para ver los resultados, gráficos y explicación clínica aquí.")
    else:
        res = st.session_state['prediccion_4modelos']
        
        st.markdown(f"### 🏥 Resultados para: **{res['atleta']}**")
        st.success("✅ Evaluación guardada exitosamente en el historial.")
        
        # Creamos 4 PESTAÑAS
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Los 4 Modelos", "🕸️ Radar", "🧬 Importancia", "🩺 Explicación Clínica"])
        
        # --- PESTAÑA 1: LOS 4 MODELOS ---
        with tab1:
            st.markdown("#### Comparativa de Algoritmos")
            
            # Creamos las 4 tarjetas visuales para cada modelo
            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)
            
            with c1:
                st.metric("Logistic Regression", f"{res['prob_log']*100:.1f}%", "Regresión Logística")
                st.progress(res['prob_log'])
            with c2:
                st.metric("Linear Regression", f"{res['prob_lin']*100:.1f}%", "Regresión Lineal")
                st.progress(res['prob_lin'])
            with c3:
                st.metric("Random Forest", f"{res['prob_rf']*100:.1f}%", "Bosques Aleatorios")
                st.progress(res['prob_rf'])
            with c4:
                st.metric("Neural Network", f"{res['prob_nn']*100:.1f}%", "Red Neuronal (MLP)")
                st.progress(res['prob_nn'])
                
            st.markdown("---")
            df_res = pd.DataFrame({
                "Modelo": ["Logistic Regression", "Linear Regression", "Random Forest", "Neural Network"],
                "Riesgo (%)": [res['prob_log']*100, res['prob_lin']*100, res['prob_rf']*100, res['prob_nn']*100]
            })
            df_res["Riesgo (%)"] = df_res["Riesgo (%)"].round(1)
            df_res["Nivel"] = df_res["Riesgo (%)"].apply(lambda x: "🔴 Alto" if x > 66 else ("🟡 Medio" if x > 33 else "🟢 Bajo"))
            
            fig_bar = px.bar(
                df_res, x="Riesgo (%)", y="Modelo", orientation="h", color="Nivel",
                color_discrete_map={"🟢 Bajo": "#2ECC71", "🟡 Medio": "#F39C12", "🔴 Alto": "#E74C3C"},
                text="Riesgo (%)"
            )
            fig_bar.update_layout(height=250, xaxis_range=[0, 100], paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- PESTAÑA 2: RADAR ---
        with tab2:
            st.markdown("#### Perfil del Atleta vs Promedio")
            prom_sanos = df_historico[df_historico[TARGET]==0][["acwr", "dolor_muscular_actual", "horas_sueno", "minutos_entrenamiento"]].mean()
            prom_lesionados = df_historico[df_historico[TARGET]==1][["acwr", "dolor_muscular_actual", "horas_sueno", "minutos_entrenamiento"]].mean()
            
            categorias = ['ACWR', 'Dolor', 'Sueño', 'Minutos']
            atleta_vals = [res['acwr']/2.5, res['dolor']/10, res['horas_sueno']/12, res['minutos']/200]
            sanos_vals = [prom_sanos['acwr']/2.5, prom_sanos['dolor_muscular_actual']/10, prom_sanos['horas_sueno']/12, prom_sanos['minutos_entrenamiento']/200]
            lesionados_vals = [prom_lesionados['acwr']/2.5, prom_lesionados['dolor_muscular_actual']/10, prom_lesionados['horas_sueno']/12, prom_lesionados['minutos_entrenamiento']/200]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=sanos_vals, theta=categorias, fill='toself', name='Promedio Sano', line_color='#2ECC71'))
            fig_radar.add_trace(go.Scatterpolar(r=lesionados_vals, theta=categorias, fill='toself', name='Promedio Lesionado', line_color='#E74C3C'))
            fig_radar.add_trace(go.Scatterpolar(r=atleta_vals, theta=categorias, fill='toself', name='Atleta Actual', line_color='#00B4D8', line_width=3))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True, height=350, margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # --- PESTAÑA 3: IMPORTANCIA ---
        with tab3:
            st.markdown("#### Top Variables Predictivas (Random Forest)")
            pre = rf.named_steps['pre']
            cat_encoder = pre.named_transformers_['cat'].named_steps['onehot']
            cat_features = cat_encoder.get_feature_names_out(['posicion', 'zona_lesion_previa'])
            
            feature_names = FEATURES_NUMERICAS + list(cat_features)
            importances = rf.named_steps['clf'].feature_importances_
            
            df_imp = pd.DataFrame({"Variable": feature_names, "Importancia": importances}).sort_values("Importancia", ascending=True).tail(10)
            
            traducciones = {
                "acwr": "ACWR (Carga)", "dolor_muscular_actual": "Dolor", "horas_sueno": "Horas Sueño",
                "calidad_sueno": "Calidad Sueño", "minutos_entrenamiento": "Minutos", 
                "rpe": "RPE", "lesiones_previas": "Lesiones Previas", "edad": "Edad", "imc": "IMC"
            }
            df_imp["Variable"] = df_imp["Variable"].apply(lambda x: traducciones.get(x, x))
            
            fig_imp = px.bar(
                df_imp, x="Importancia", y="Variable", orientation="h",
                color="Importancia", color_continuous_scale="Blues", text_auto=".3f"
            )
            fig_imp.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_imp, use_container_width=True)

        # --- PESTAÑA 4: EXPLICACIÓN CLÍNICA ---
        with tab4:
            st.markdown("#### 🩺 Diagnóstico y Justificación")
            st.write(f"El modelo **Random Forest** determinó un riesgo **{res['nivel_final']}** ({res['prob_rf']*100:.1f}%). Aquí está el porqué:")
            
            # Generar explicaciones basadas en los datos
            if res['acwr'] > 1.5:
                st.error(f"🚨 **ACWR Elevado ({res['acwr']}):** La carga de la última semana es mucho mayor a la habitual. El umbral seguro es 1.5. Hay sobrecarga aguda.")
            elif res['acwr'] > 1.3:
                st.warning(f"⚠️ **ACWR en zona de alerta ({res['acwr']}):** La carga está subiendo rápido. Vigilar evolución.")
            else:
                st.success(f"✅ **ACWR Óptimo ({res['acwr']}):** La carga está equilibrada y asimilable.")

            if res['dolor'] >= 6:
                st.error(f"🚨 **Dolor Muscular Alto ({res['dolor']}/10):** El atleta ya reporta molestias fuertes antes de la lesión. Es un indicador crítico.")
            elif res['dolor'] >= 3:
                st.warning(f"⚠️ **Dolor Moderado ({res['dolor']}/10):** Molestias presentes. Revisar zonas de hombro y rodilla.")
            
            if res['horas_sueno'] < 6:
                st.error(f"🚨 **Déficit de Sueño ({res['horas_sueno']} h):** Dormir menos de 6 horas eleva la fatiga neuromuscular y multiplica el riesgo.")
            elif res['horas_sueno'] < 7:
                st.warning(f"⚠️ **Sueño Insuficiente ({res['horas_sueno']} h):** Idealmente el atleta debe dormir 7-8 horas para recuperarse.")

            if res['calidad_sueno'] <= 2:
                st.error("🚨 **Mala Calidad de Sueño:** Indica que el descanso no fue reparador (posible fatiga acumulada).")
                
            if res['lesiones_previas'] > 0:
                st.warning(f"⚠️ **Antecedentes de Lesión ({res['lesiones_previas']}):** Tener lesiones previas es el factor de riesgo más fuerte a futuro. Zonas vulnerables.")

            st.info(
                "**Recomendación del Sistema:**\n"
                + ("Suspender carga de alto impacto por 48h y derivar a fisioterapia inmediatamente." if res['nivel_final'] == "ALTO" else 
                   ("Reducir 20-30% el volumen de saltos y monitorear el dolor diariamente." if res['nivel_final'] == "MEDIO" else 
                    "Continuar con el plan de entrenamiento habitual sin riesgos."))
            )