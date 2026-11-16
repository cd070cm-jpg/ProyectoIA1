# diagnostico.py
import sys, pickle
from pathlib import Path

print("=== PYTHON PATH ===")
for p in sys.path[:5]:
    print(" ", p)

print("\n=== ARCHIVOS utils.py EN EL PROYECTO ===")
for f in Path(".").rglob("utils.py"):
    print(" ", f.resolve())

print("\n=== ARCHIVOS .pkl EN EL PROYECTO ===")
for f in Path(".").rglob("*.pkl"):
    print(" ", f.resolve(), f"({f.stat().st_size} bytes)")
    try:
        with open(f, "rb") as fh:
            obj = pickle.load(fh)
        print("    tipo:", type(obj).__name__)
        if isinstance(obj, dict):
            print("    claves:", list(obj.keys()))
    except Exception as e:
        print("    ERROR al cargar:", e)

print("\n=== QUÉ DEVUELVE cargar_modelo() ===")
try:
    from app.utils import cargar_modelo
    artefacto = cargar_modelo()
    print("tipo devuelto:", type(artefacto).__name__)
    if isinstance(artefacto, dict):
        print("claves:", list(artefacto.keys()))
    else:
        print("⚠️ NO es dict -> estás cargando el modelo viejo")
except Exception as e:
    import traceback; traceback.print_exc()