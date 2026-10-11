from __future__ import annotations

from pathlib import Path

import pandas as pd

from recomendaciones import calcular_rachas, clasificar_fila

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "dataset" / "atalaya_lecturas.csv"


def main() -> None:
    if not CSV_PATH.exists():
        print(f"No existe el dataset: {CSV_PATH}")
        return
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        print("El dataset esta vacio.")
        return
    print("\n=== ATALAYA: DATASET ===")
    print(f"Filas: {len(df)}")
    print("\n--- Por etiqueta binaria ---")
    print(df["etiqueta"].value_counts().sort_index().to_string())
    print("\n--- Por sesion ---")
    print(df["sesion"].value_counts().to_string())
    datos = df.copy()
    for col in ["etiqueta", "temp_c", "humedad_pct", "pir", "luz"]:
        datos[col] = pd.to_numeric(datos[col], errors="coerce")
    datos = datos.dropna(subset=["etiqueta", "temp_c", "humedad_pct", "pir", "luz"])
    datos["pir"] = datos["pir"].astype(int)
    datos["luz"] = datos["luz"].astype(int)
    datos = calcular_rachas(datos)
    datos["clase"] = datos.apply(lambda r: clasificar_fila(r["etiqueta"], r["temp_c"], r["pir"], r["luz"], r["racha_pir"], r["racha_luz"]), axis=1)
    print("\n--- Clases para IA y recomendaciones ---")
    print(datos["clase"].value_counts().to_string())
    insuficientes = datos["clase"].value_counts()
    insuficientes = insuficientes[insuficientes < 6]
    if not insuficientes.empty:
        print(f"\nFaltan ejemplos para entrenar 70/15/15: {insuficientes.to_dict()}")
        print("Graba casos breves y sostenidos de movimiento y poca luz.")
    else:
        print("\nDataset apto para entrenar: .\\entrenar_modelo.ps1")


if __name__ == "__main__":
    main()
