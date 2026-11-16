import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

st.set_page_config(page_title="Gestión de Atletas", page_icon="🏃", layout="wide")
st.title("🏃 Gestión de Atletas (UPDS Voleibol)")
st.markdown("Registra los datos base de los jugadores para luego evaluar su carga física.")

DB_PATH = Path(__file__).resolve().parent.parent.parent / "sistema_lesiones.db"

# Formulario de registro
with st.expander("➕ Registrar Nuevo Atleta", expanded=False):
    with st.form("form_nuevo_atleta"):
        col1, col2, col3 = st.columns(3)
        cedula = col1.text_input("Cédula de Identidad")
        nombre = col2.text_input("Nombre Completo")
        edad = col3.number_input("Edad", 15, 35, 20)
        
        col4, col5, col6 = st.columns(3)
        posicion = col4.selectbox("Posición", ["acomodador", "opuesto", "central", "receptor", "libero"])
        estatura = col5.number_input("Estatura (cm)", 140.0, 220.0, 175.0, step=0.5)
        peso = col6.number_input("Peso (kg)", 40.0, 120.0, 70.0, step=0.5)
        
        submit = st.form_submit_button("Guardar Atleta", type="primary")
        
        if submit:
            if not cedula or not nombre:
                st.error("La cédula y el nombre son obligatorios.")
            else:
                try:
                    with sqlite3.connect(DB_PATH) as conn:
                        conn.execute("""
                            INSERT INTO atletas (cedula, nombre, edad, posicion, estatura, peso)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (cedula, nombre, edad, posicion, estatura, peso))
                    st.success(f"✅ Atleta {nombre} registrado exitosamente.")
                except sqlite3.IntegrityError:
                    st.error("❌ Error: Ya existe un atleta con esa cédula.")

st.markdown("---")
st.subheader("📋 Lista de Atletas Registrados")

# Mostrar tabla
try:
    with sqlite3.connect(DB_PATH) as conn:
        df_atletas = pd.read_sql_query("SELECT id, cedula, nombre, edad, posicion, estatura, peso, fecha_registro FROM atletas ORDER BY nombre", conn)
    
    if df_atletas.empty:
        st.info("No hay atletas registrados aún. Usa el formulario de arriba.")
    else:
        df_atletas["IMC"] = (df_atletas["peso"] / (df_atletas["estatura"]/100)**2).round(1)
        st.dataframe(df_atletas.drop(columns=["id", "fecha_registro"]), use_container_width=True, hide_index=True)
except Exception as e:
    st.error(f"Error al cargar atletas: {e}")