from __future__ import annotations

import argparse
import copy
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from recomendaciones import CLASES, FEATURES, RECOMENDACIONES, calcular_rachas, clasificar_fila

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "dataset" / "atalaya_lecturas.csv"
DEFAULT_MODEL = BASE_DIR / "modelos" / "atalaya_modelo.joblib"
DEFAULT_REPORT = BASE_DIR / "reportes" / "metricas_modelos.md"
DEFAULT_CURVE = BASE_DIR / "reportes" / "curva_aprendizaje_red.png"
TARGET = "clase"
RANDOM_STATE = 42
MINIMO_POR_CLASE = 6


def crear_preprocesador() -> ColumnTransformer:
    return ColumnTransformer([("numericas", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), FEATURES)])


def crear_modelos_base() -> dict[str, Pipeline]:
    return {
        "regresion_logistica": Pipeline([("prep", crear_preprocesador()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "arbol_decision": Pipeline([
            ("prep", ColumnTransformer([("numericas", SimpleImputer(strategy="median"), FEATURES)])),
            ("model", DecisionTreeClassifier(max_depth=6, min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }


def preparar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    requeridas = ["sesion", "timestamp", "etiqueta", "temp_c", "humedad_pct", "pir", "luz"]
    faltantes = [col for col in requeridas if col not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el CSV: {faltantes}")
    df = df.copy()
    for col in ["etiqueta", "temp_c", "humedad_pct", "pir", "luz"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["etiqueta", "temp_c", "humedad_pct", "pir", "luz"])
    df["etiqueta"] = df["etiqueta"].astype(int)
    df["pir"] = df["pir"].astype(int)
    df["luz"] = df["luz"].astype(int)
    df = calcular_rachas(df)
    df[TARGET] = df.apply(lambda r: clasificar_fila(r["etiqueta"], r["temp_c"], r["pir"], r["luz"], r["racha_pir"], r["racha_luz"]), axis=1)
    if df[TARGET].nunique() < 2:
        raise ValueError("Necesitas registros normales y anomalos para entrenar.")
    return df


def medir(modelo: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, object]:
    pred = modelo.predict(x)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, average="macro", zero_division=0),
        "recall": recall_score(y, pred, average="macro", zero_division=0),
        "f1": f1_score(y, pred, average="macro", zero_division=0),
        "matriz": confusion_matrix(y, pred, labels=sorted(y.unique())),
        "etiquetas_matriz": sorted(y.unique()),
        "reporte": classification_report(y, pred, zero_division=0),
    }


def entrenar_red(x_train, y_train, x_val, y_val, clases_presentes: list[str]):
    preprocesador = crear_preprocesador()
    x_train_t = preprocesador.fit_transform(x_train)
    x_val_t = preprocesador.transform(x_val)
    red = MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu", solver="adam", learning_rate_init=0.01, alpha=0.001, max_iter=1, random_state=RANDOM_STATE)
    perdida_train, perdida_val, exactitud_train, exactitud_val = [], [], [], []
    mejor_red, mejor_f1, sin_mejora = None, -1.0, 0
    for epoca in range(1, 301):
        if epoca == 1:
            red.partial_fit(x_train_t, y_train, classes=clases_presentes)
        else:
            red.partial_fit(x_train_t, y_train)
        pred_train, pred_val = red.predict(x_train_t), red.predict(x_val_t)
        perdida_train.append(float(red.loss_))
        perdida_val.append(log_loss(y_val, red.predict_proba(x_val_t), labels=red.classes_))
        exactitud_train.append(accuracy_score(y_train, pred_train))
        exactitud_val.append(accuracy_score(y_val, pred_val))
        f1_val = f1_score(y_val, pred_val, average="macro", zero_division=0)
        if f1_val > mejor_f1:
            mejor_f1, mejor_red, sin_mejora = f1_val, copy.deepcopy(red), 0
        else:
            sin_mejora += 1
            if sin_mejora >= 30:
                break
    assert mejor_red is not None
    return Pipeline([("prep", preprocesador), ("model", mejor_red)]), perdida_train, perdida_val, exactitud_train, exactitud_val


def guardar_curva(path: Path, perdida_train, perdida_val, exactitud_train, exactitud_val) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epocas = range(1, len(perdida_train) + 1)
    fig, ejes = plt.subplots(1, 2, figsize=(10, 4))
    ejes[0].plot(epocas, perdida_train, label="Entrenamiento")
    ejes[0].plot(epocas, perdida_val, label="Validacion")
    ejes[0].set(title="Perdida por epoca", xlabel="Epoca", ylabel="Perdida")
    ejes[0].legend(); ejes[0].grid(alpha=0.3)
    ejes[1].plot(epocas, exactitud_train, label="Entrenamiento")
    ejes[1].plot(epocas, exactitud_val, label="Validacion")
    ejes[1].set(title="Exactitud por epoca", xlabel="Epoca", ylabel="Exactitud", ylim=(0, 1.05))
    ejes[1].legend(); ejes[1].grid(alpha=0.3)
    fig.suptitle("Atalaya: aprendizaje de la red neuronal multiclase")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def escribir_reporte(path: Path, total: int, tamanos: dict[str, int], clases_por_conjunto: dict[str, dict], validacion: dict[str, dict], prueba: dict, curva: Path) -> None:
    lineas = ["# Resultados de IA supervisada - Atalaya", "", "## Clases y recomendaciones", "", "| Clase | Recomendacion |", "|---|---|"]
    lineas += [f"| {clase} | {RECOMENDACIONES[clase]} |" for clase in CLASES]
    lineas += ["", "## Estructura experimental", "", f"Filas usadas: {total}", "", "| Conjunto | Proporcion | Filas | Distribucion de clases |", "|---|---:|---:|---|"]
    for nombre, proporcion in [("Entrenamiento", "70%"), ("Validacion", "15%"), ("Prueba", "15%")]:
        distribucion = ", ".join(f"{clase}={n}" for clase, n in clases_por_conjunto[nombre].items())
        lineas.append(f"| {nombre} | {proporcion} | {tamanos[nombre]} | {distribucion} |")
    lineas += ["", "La red se ajusta con entrenamiento. La validacion selecciona la mejor epoca mediante F1 macro. La prueba se usa una sola vez para la medicion final.", "", "## Comparacion en validacion", "", "| Modelo | Accuracy | Precision macro | Recall macro | F1 macro |", "|---|---:|---:|---:|---:|"]
    for nombre, resultado in validacion.items():
        lineas.append(f"| {nombre} | {resultado['accuracy']:.3f} | {resultado['precision']:.3f} | {resultado['recall']:.3f} | {resultado['f1']:.3f} |")
    lineas += ["", "## Modelo usado", "", "La prediccion usa una red neuronal MLP con dos capas ocultas: 16 y 8 neuronas. Regresion logistica y arbol de decision se conservan como comparacion.", "", "## Resultado final en prueba", "", "| Accuracy | Precision macro | Recall macro | F1 macro |", "|---:|---:|---:|---:|", f"| {prueba['accuracy']:.3f} | {prueba['precision']:.3f} | {prueba['recall']:.3f} | {prueba['f1']:.3f} |", "", f"Matriz de confusion. Orden: {prueba['etiquetas_matriz']}", "", "```text", str(prueba["matriz"]), "```", "", "```text", str(prueba["reporte"]), "```", "", f"Curvas: `{curva.name}`."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lineas), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena la red neuronal multiclase de Atalaya.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--curve-out", type=Path, default=DEFAULT_CURVE)
    args = parser.parse_args()
    df = preparar_dataset(pd.read_csv(args.csv))
    x, y = df[FEATURES], df[TARGET]
    conteo = y.value_counts()
    escasas = conteo[conteo < MINIMO_POR_CLASE]
    if not escasas.empty:
        raise ValueError(f"Faltan al menos {MINIMO_POR_CLASE} ejemplos por clase para dividir 70/15/15: {escasas.to_dict()}.")
    clases_presentes = sorted(y.unique(), key=lambda clase: CLASES.index(clase))
    x_train, x_temporal, y_train, y_temporal = train_test_split(x, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)
    x_val, x_test, y_val, y_test = train_test_split(x_temporal, y_temporal, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temporal)
    validacion = {}
    for nombre, modelo in crear_modelos_base().items():
        modelo.fit(x_train, y_train)
        validacion[nombre] = medir(modelo, x_val, y_val)
    red, p_train, p_val, a_train, a_val = entrenar_red(x_train, y_train, x_val, y_val, clases_presentes)
    validacion["red_neuronal"] = medir(red, x_val, y_val)
    prueba = medir(red, x_test, y_test)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(red, args.model_out)
    guardar_curva(args.curve_out, p_train, p_val, a_train, a_val)
    tamanos = {"Entrenamiento": len(x_train), "Validacion": len(x_val), "Prueba": len(x_test)}
    conjuntos = {"Entrenamiento": y_train.value_counts().to_dict(), "Validacion": y_val.value_counts().to_dict(), "Prueba": y_test.value_counts().to_dict()}
    escribir_reporte(args.report_out, len(df), tamanos, conjuntos, validacion, prueba, args.curve_out)
    print(f"Clases: {clases_presentes}")
    print(f"Modelo guardado en: {args.model_out}")
    print(f"F1 macro final: {prueba['f1']:.3f}")


if __name__ == "__main__":
    main()
