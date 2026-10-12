# Atalaya: sistema de recomendaciones con IA

## Que hace el proyecto

Atalaya es un nodo ESP32 que mide temperatura, humedad, movimiento y luz. Envia cada lectura por medio de WiFi al computador. El computador usa una red neuronal para asignar una clase y devolver una recomendación concreta al OLED.

```text
Sensores -> ESP32 -> WiFi -> servidor Python -> red neuronal -> clase y recomendacion -> OLED
```

El PC y ESP32 deben estar en la misma red WiFi de 2.4 GHz.

## Clases y recomendaciones

| Clase | Recomendacion |
|---|---|
| NORMAL | Todo en orden. No se requiere accion. |
| TEMP_ALTA_LEVE | Verificar ventilacion. |
| TEMP_ALTA_CRITICA | Revisar el equipo de inmediato. |
| MOVIMIENTO_LEVE | Verificar si el movimiento fue autorizado. |
| MOVIMIENTO_CRITICO | Revisar posible presencia no autorizada. |
| POCA_LUZ_LEVE | Monitorear la iluminacion. |
| POCA_LUZ_CRITICA | Revisar iluminacion. |

Movimiento o poca luz son leves durante menos de cinco lecturas seguidas. Desde cinco lecturas continuas se consideran criticos.

## La red neuronal y el perceptron

La IA usada es una red neuronal tipo **MLP** (perceptron multicapa). Tiene:

- Seis entradas: temperatura, humedad, PIR, luz, racha del PIR y racha de luz.
- Dos capas ocultas: 16 neuronas y 8 neuronas.
- Una salida multiclase: una de las siete clases anteriores.
- Funcion de activacion ReLU en las capas ocultas.
- Entrenamiento por epocas con el algoritmo Adam.

Un perceptron recibe valores, los multiplica por pesos aprendidos y produce una salida. La red conecta muchos perceptrones: las capas ocultas aprenden patrones como temperatura alta o movimiento sostenido. La ultima capa decide la clase con mayor probabilidad.

La recomendacion : cada clase predicha tiene una recomendacion fija, verificable y adecuada al evento detectado.

## Evaluacion correcta

El dataset se divide de forma estratificada:

- 70% entrenamiento.
- 15% validacion. Escoge la mejor epoca y compara modelos.
- 15% prueba final. Mide el rendimiento una sola vez.

El reporte incluye Accuracy, Precision, Recall, F1 macro, matriz de confusion y la curva de perdida/exactitud. F1 macro es importante porque da el mismo peso a las clases poco frecuentes y a NORMAL.

## Archivos principales

- `firmware_esp32_ia/firmware_esp32_ia.ino`: firmware para cargar en la ESP32.
- `firmware_esp32_ia/config_local.example.h`: plantilla para configurar WiFi y URL local sin subir claves.
- `receiver.py`: servidor que recibe datos, calcula rachas y devuelve la recomendacion.
- `recomendaciones.py`: clases, umbrales y textos de recomendacion.
- `dataset/atalaya_lecturas.csv`: datos recolectados. No se sube a GitHub.
- `dataset/ejemplo_lecturas.csv`: ejemplo seguro del formato del dataset.
- `train_classifier.py`: entrenamiento y evaluacion de la red.
- `modelos/atalaya_modelo.joblib`: se crea despues de entrenar la red neuronal.
- `reportes/metricas_modelos.md`: se crea despues de entrenar y guarda los resultados.
- `reportes/curva_aprendizaje_red.png`: se crea despues de entrenar y muestra perdida/exactitud.
- `docs/cronograma_presupuesto_control.md`: cronograma, presupuesto y propuesta de control en Notion.
- `docs/notion_tareas.csv`: tareas importables a Notion.

## Configurar WiFi 

1. Copia `firmware_esp32_ia/config_local.example.h`.
2. Renombra la copia como `firmware_esp32_ia/config_local.h`.
3. Escribe tu SSID, password y URL del servidor local.

`config_local.h` esta en `.gitignore`,

## Paso siguiente: preparar datos para las siete clases

Los datos actuales aun no tienen suficientes ejemplos de `POCA_LUZ_LEVE` y `MOVIMIENTO_CRITICO`. Antes de capturar nuevos datos, guarda una copia manual del CSV si deseas conservar el experimento binario anterior.

Luego ejecuta una vez:

```powershell
.\reiniciar_dataset.ps1
```

Graba datos con estos patrones:

1. Normal: sensores sin eventos anormales.
2. Temperatura: deja algunos segundos entre 36 y 39 C y otros por encima de 39 C.
3. Movimiento: alterna gestos de 1 a 2 segundos y movimiento continuo de 5 a 10 segundos.
4. Luz: alterna tapar el LDR 1 a 2 segundos y mantenerlo tapado 5 a 10 segundos.

Usa los mismos comandos de siempre:

```powershell
.\grabar_normal.ps1
.\grabar_anomalia_temp.ps1
.\grabar_anomalia_movimiento.ps1
.\grabar_anomalia_luz.ps1
```

Revisa el balance antes de entrenar:

```powershell
.\leer_dataset.ps1
```

Necesitas al menos seis ejemplos de cada clase presente. Es mejor registrar varias decenas de cada una.

## Por que puede salir `Sin modelo cargado`

Ese mensaje significa que el servidor recibe datos de la ESP32, pero todavia no tiene una red multiclase valida para predecir. Puede pasar por dos razones:

- Aun no existe `modelos/atalaya_modelo.joblib`.
- El archivo existe, pero es un modelo viejo binario de 0/1 y no sirve para las siete clases actuales.

La solucion es completar los datos faltantes y entrenar de nuevo:

```powershell
.\leer_dataset.ps1
.\entrenar_modelo.ps1
```

Cuando el entrenamiento termine bien, `prediccion_en_vivo.ps1` ya debe mostrar `clase=NORMAL`, `clase=MOVIMIENTO_CRITICO` u otra clase real.

## Entrenar y demostrar

```powershell
.\entrenar_modelo.ps1
.\prediccion_en_vivo.ps1
```

Despues de entrenar, vuelve a cargar `firmware_esp32_ia.ino` en Arduino IDE. En el monitor serial debe aparecer `clase_ia=` y una recomendacion. En el OLED aparecera la version corta.

## Red de celular para la exposicion

La ESP32 solo funciona con WiFi de 2.4 GHz. Conecta el PC y la ESP32 al mismo hotspot del celular. Luego ejecuta:

```powershell
.\diagnosticar_red.ps1
```

Copia la URL mostrada en `SERVER_URL` dentro del firmware, vuelve a cargar la ESP32 y abre `prediccion_en_vivo.ps1`.
