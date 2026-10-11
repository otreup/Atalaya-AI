"""Clases con severidad y recomendaciones de Atalaya."""

from __future__ import annotations

TEMP_UMBRAL = 36.0
TEMP_CRITICA = 39.0
RACHA_CRITICA = 5

FEATURES = ["temp_c", "humedad_pct", "pir", "luz", "racha_pir", "racha_luz"]

CLASES = [
    "NORMAL",
    "TEMP_ALTA_LEVE",
    "TEMP_ALTA_CRITICA",
    "MOVIMIENTO_LEVE",
    "MOVIMIENTO_CRITICO",
    "POCA_LUZ_LEVE",
    "POCA_LUZ_CRITICA",
]

RECOMENDACIONES = {
    "NORMAL": "Todo en orden. No se requiere accion.",
    "TEMP_ALTA_LEVE": "Temperatura elevada leve. Verificar ventilacion.",
    "TEMP_ALTA_CRITICA": "Temperatura critica. Revisar el equipo de inmediato.",
    "MOVIMIENTO_LEVE": "Movimiento puntual. Verificar si fue autorizado.",
    "MOVIMIENTO_CRITICO": "Movimiento sostenido. Revisar posible presencia no autorizada.",
    "POCA_LUZ_LEVE": "Disminucion breve de luz. Monitorear.",
    "POCA_LUZ_CRITICA": "Oscuridad sostenida. Revisar iluminacion.",
}

RECOMENDACIONES_OLED = {
    "NORMAL": "Todo normal",
    "TEMP_ALTA_LEVE": "Temp alta: ventilar",
    "TEMP_ALTA_CRITICA": "TEMP CRITICA: revisar",
    "MOVIMIENTO_LEVE": "Movim. puntual",
    "MOVIMIENTO_CRITICO": "INTRUSION posible",
    "POCA_LUZ_LEVE": "Poca luz breve",
    "POCA_LUZ_CRITICA": "OSCURIDAD sostenida",
}


def clasificar_fila(etiqueta: int, temp_c: float, pir: int, luz: int, racha_pir: int, racha_luz: int) -> str:
    """Crea la etiqueta de entrenamiento a partir de la lectura original."""
    if etiqueta == 0:
        return "NORMAL"
    if temp_c >= TEMP_CRITICA:
        return "TEMP_ALTA_CRITICA"
    if temp_c >= TEMP_UMBRAL:
        return "TEMP_ALTA_LEVE"
    if pir == 1:
        return "MOVIMIENTO_CRITICO" if racha_pir >= RACHA_CRITICA else "MOVIMIENTO_LEVE"
    if luz == 1:
        return "POCA_LUZ_CRITICA" if racha_luz >= RACHA_CRITICA else "POCA_LUZ_LEVE"
    return "NORMAL"


def calcular_rachas(df):
    """Calcula lecturas consecutivas activas de PIR y LDR por sesion."""
    df = df.sort_values(["sesion", "timestamp"]).copy()

    def racha(serie):
        cambios = (serie != serie.shift()).cumsum()
        return serie.groupby(cambios).cumcount().add(1).mul(serie)

    df["racha_pir"] = df.groupby("sesion")["pir"].transform(racha)
    df["racha_luz"] = df.groupby("sesion")["luz"].transform(racha)
    return df
