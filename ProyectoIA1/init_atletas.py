# init_atletas.py
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "sistema_lesiones.db"

with sqlite3.connect(DB_PATH) as conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS atletas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cedula TEXT UNIQUE NOT NULL,
            nombre TEXT NOT NULL,
            edad INTEGER,
            posicion TEXT,
            estatura REAL,
            peso REAL,
            fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
print("✅ Tabla 'atletas' lista en la base de datos.")