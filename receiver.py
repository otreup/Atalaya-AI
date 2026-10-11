from __future__ import annotations

import argparse
import csv
import json
import socket
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from recomendaciones import FEATURES, RECOMENDACIONES, RECOMENDACIONES_OLED

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "dataset" / "atalaya_lecturas.csv"
FIELDNAMES = ["timestamp", "sesion", "etiqueta", "temp_c", "humedad_pct", "pir", "luz", "racha_pir", "racha_luz", "estado_local", "clase_predicha", "recomendacion", "cliente_ip"]


def cargar_modelo(model_path: Path | None) -> Any | None:
    if model_path is None or not model_path.exists():
        if model_path is not None:
            print(f"No existe el modelo: {model_path}")
        return None
    try:
        import joblib
        modelo = joblib.load(model_path)
        prep = modelo.named_steps.get("prep")
        columnas = prep.transformers_[0][2] if prep is not None else []
        if list(columnas) != FEATURES:
            print("El modelo guardado es binario/anterior.")
            print(f"Columnas del modelo: {list(columnas)}")
            print(f"Columnas esperadas: {FEATURES}")
            print("Entrena el modelo multiclase antes de usar prediccion en vivo.")
            return None
        clases = list(getattr(modelo.named_steps.get("model"), "classes_", []))
        if not clases or all(str(clase) in {"0", "1"} for clase in clases):
            print("El modelo cargado parece binario. Entrena el modelo multiclase antes de usar prediccion en vivo.")
            return None
        return modelo
    except Exception as exc:
        print(f"No se pudo cargar el modelo: {exc}")
        return None


def predecir(modelo: Any | None, fila: dict[str, Any]) -> str | None:
    if modelo is None:
        return None
    try:
        import pandas as pd
        entrada = pd.DataFrame([{col: fila[col] for col in FEATURES}])
        return str(modelo.predict(entrada)[0])
    except Exception as exc:
        print(f"No se pudo predecir esta lectura: {exc}")
        return None


def asegurar_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as archivo:
            csv.DictWriter(archivo, fieldnames=FIELDNAMES).writeheader()
        return
    with path.open(newline="", encoding="utf-8-sig") as archivo:
        encabezados = next(csv.reader(archivo), [])
    if encabezados != FIELDNAMES:
        raise ValueError("El CSV usa el esquema anterior. Ejecuta .\\reiniciar_dataset.ps1 para iniciar el dataset multiclase.")


def guardar_fila(path: Path, fila: dict[str, Any]) -> None:
    asegurar_csv(path)
    with path.open("a", newline="", encoding="utf-8") as archivo:
        csv.DictWriter(archivo, fieldnames=FIELDNAMES).writerow(fila)


class RachaTracker:
    def __init__(self) -> None:
        self.pir = 0
        self.luz = 0

    def actualizar(self, pir: int, luz: int) -> tuple[int, int]:
        self.pir = self.pir + 1 if pir == 1 else 0
        self.luz = self.luz + 1 if luz == 1 else 0
        return self.pir, self.luz


def crear_handler(args: argparse.Namespace, modelo: Any | None, rachas: RachaTracker):
    class AtalayaHandler(BaseHTTPRequestHandler):
        def responder(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/status":
                self.responder(200, {"ok": True, "modelo_cargado": modelo is not None, "guardando_csv": not args.no_save})
            else:
                self.responder(404, {"ok": False, "error": "ruta no encontrada"})

        def do_POST(self) -> None:
            if self.path != "/api/lectura":
                self.responder(404, {"ok": False, "error": "ruta no encontrada"})
                return
            try:
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8")
                data = json.loads(raw)
                pir, luz = int(data.get("pir", 0)), int(data.get("luz", 0))
                racha_pir, racha_luz = rachas.actualizar(pir, luz)
                fila_modelo = {"temp_c": float(data["temp_c"]), "humedad_pct": float(data["humedad_pct"]), "pir": pir, "luz": luz, "racha_pir": racha_pir, "racha_luz": racha_luz}
                clase = predecir(modelo, fila_modelo)
                recomendacion = RECOMENDACIONES.get(clase, "") if clase else ""
                oled = RECOMENDACIONES_OLED.get(clase, "") if clase else ""
                fila = {"timestamp": datetime.now().isoformat(timespec="seconds"), "sesion": args.sesion, "etiqueta": args.label, **fila_modelo, "estado_local": data.get("estado_local", ""), "clase_predicha": clase or "", "recomendacion": recomendacion, "cliente_ip": self.client_address[0]}
                if not args.no_save:
                    guardar_fila(args.csv, fila)
                etiqueta = "sin_guardar" if args.no_save else str(args.label)
                print(f"[{fila['timestamp']}] label={etiqueta} temp={fila['temp_c']} pir={pir}(r={racha_pir}) luz={luz}(r={racha_luz}) clase={clase or '-'}")
                self.responder(200, {"ok": True, "clase": clase, "recomendacion": recomendacion, "recomendacion_oled": oled})
            except Exception as exc:
                self.responder(400, {"ok": False, "error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return AtalayaHandler


def obtener_ip_local() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Servidor local para Atalaya.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--label", type=int, choices=[0, 1], required=True)
    parser.add_argument("--sesion", required=True)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    if not args.no_save:
        asegurar_csv(args.csv)
    modelo = cargar_modelo(args.model)
    ip = obtener_ip_local()
    print("Servidor Atalaya listo.")
    print(f"URL ESP32: http://{ip}:{args.port}/api/lectura" if ip else "No se detecto la IP local.")
    print("Modo prediccion: no se guardara en CSV." if args.no_save else f"Guardando en: {args.csv}")
    print("Modelo cargado: clase y recomendacion activas." if modelo else "Sin modelo cargado.")
    HTTPServer((args.host, args.port), crear_handler(args, modelo, RachaTracker())).serve_forever()


if __name__ == "__main__":
    main()
