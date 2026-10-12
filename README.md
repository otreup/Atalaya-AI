# Atalaya: An AI-Based Recommendation System

## What the Project Does

Atalaya is an ESP32-based device that measures temperature, humidity, movement, and light. It sends each reading to a computer through Wi-Fi. The computer uses a neural network to classify the readings and send a specific recommendation back to the OLED display.

```text
Sensors -> ESP32 -> Wi-Fi -> Python Server -> Neural Network -> Class and Recommendation -> OLED
```

The computer and the ESP32 must be connected to the same 2.4 GHz Wi-Fi network.

## Classes and Recommendations

| Class              | Recommendation                                         |
| ------------------ | ------------------------------------------------------ |
| NORMAL             | Everything is working normally. No action is required. |
| TEMP_ALTA_LEVE     | Check the ventilation.                                 |
| TEMP_ALTA_CRITICA  | Check the equipment immediately.                       |
| MOVIMIENTO_LEVE    | Check whether the movement was authorized.             |
| MOVIMIENTO_CRITICO | Check for possible unauthorized presence.              |
| POCA_LUZ_LEVE      | Monitor the lighting conditions.                       |
| POCA_LUZ_CRITICA   | Check the lighting system.                             |

Movement and low-light events are classified as minor if they last for fewer than five consecutive readings. Starting from five consecutive readings, they are classified as critical.

## The Neural Network and the Perceptron

The AI model used in this project is a **Multilayer Perceptron (MLP)** neural network. It has:

* Six input features: temperature, humidity, PIR sensor readings, light level, consecutive PIR readings, and consecutive low-light readings.
* Two hidden layers: one with 16 neurons and another with 8 neurons.
* A multiclass output: one of the seven classes listed above.
* The ReLU activation function in the hidden layers.
* Training over multiple epochs using the Adam optimization algorithm.

A perceptron receives input values, multiplies them by learned weights, and produces an output. A neural network connects many perceptrons to learn patterns, such as high temperatures or continuous movement. The final layer selects the class with the highest predicted probability.

Each predicted class has a fixed recommendation. These recommendations are designed to be clear, verifiable, and appropriate for the detected event.

## Proper Model Evaluation

The dataset is divided using stratified sampling:

* **70% for training:** The neural network learns from these data.
* **15% for validation:** This set helps select the best training epoch and compare models.
* **15% for final testing:** This set measures the model's performance only once after model selection.

The evaluation report includes Accuracy, Precision, Recall, Macro F1-score, a confusion matrix, and learning curves for loss and accuracy.

The Macro F1-score is important because it gives equal weight to each class, including less frequent classes and the NORMAL class.

## Main Project Files

* `firmware_esp32_ia/firmware_esp32_ia.ino`: Firmware to upload to the ESP32.
* `firmware_esp32_ia/config_local.example.h`: Template for configuring Wi-Fi and the local server URL without exposing credentials.
* `receiver.py`: Server that receives sensor data, calculates consecutive readings, and returns recommendations.
* `recomendaciones.py`: Contains the classes, thresholds, and recommendation messages.
* `dataset/atalaya_lecturas.csv`: Collected data. This file is not uploaded to GitHub.
* `dataset/ejemplo_lecturas.csv`: Safe example of the dataset format.
* `train_classifier.py`: Script for training and evaluating the neural network.
* `modelos/atalaya_modelo.joblib`: File created after training the neural network.
* `reportes/metricas_modelos.md`: File created after training that stores the evaluation results.
* `reportes/curva_aprendizaje_red.png`: File created after training that shows the loss and accuracy curves.
* `docs/cronograma_presupuesto_control.md`: Project schedule, budget, and control proposal for Notion.
* `docs/notion_tareas.csv`: Task list that can be imported into Notion.

## Wi-Fi Configuration

1. Copy `firmware_esp32_ia/config_local.example.h`.
2. Rename the copy to `firmware_esp32_ia/config_local.h`.
3. Enter your Wi-Fi SSID, password, and local server URL.

The `config_local.h` file is included in `.gitignore` to prevent your local credentials from being uploaded to GitHub.

## Next Step: Preparing Data for the Seven Classes

The current dataset does not yet contain enough examples of `POCA_LUZ_LEVE` and `MOVIMIENTO_CRITICO`.

Before collecting new data, manually save a copy of the CSV file if you want to preserve the previous binary classification experiment.

Then, run the following command once:

```powershell
.\reiniciar_dataset.ps1
```

Collect data using the following patterns:

1. **Normal conditions:** Keep the sensors operating without abnormal events.
2. **Temperature:** Allow the temperature to remain between 36 and 39 °C for a few seconds in some trials, and above 39 °C in others.
3. **Movement:** Alternate between movements lasting 1–2 seconds and continuous movement lasting 5–10 seconds.
4. **Light:** Cover the LDR sensor for 1–2 seconds in some trials, and keep it covered for 5–10 seconds in others.

Use the same commands as before:

```powershell
.\grabar_normal.ps1
.\grabar_anomalia_temp.ps1
.\grabar_anomalia_movimiento.ps1
.\grabar_anomalia_luz.ps1
```

Check the dataset balance before training:

```powershell
.\leer_dataset.ps1
```

You need at least six examples for every class present in the dataset. However, collecting several dozen examples per class is recommended.

## Why the Message `Sin modelo cargado` May Appear

This message means that the server is receiving data from the ESP32, but it does not have a valid multiclass neural network available to make predictions.

This can happen for two reasons:

* The file `modelos/atalaya_modelo.joblib` does not exist yet.
* The file exists, but it contains an old binary model that classifies data as 0 or 1 and cannot be used for the current seven-class system.

To solve this problem, collect the missing data and train the model again:

```powershell
.\leer_dataset.ps1
.\entrenar_modelo.ps1
```

After successful training, `prediccion_en_vivo.ps1` should display a real class, such as `clase=NORMAL`, `clase=MOVIMIENTO_CRITICO`, or another valid class.

## Training and Demonstrating the System

Run the following commands:

```powershell
.\entrenar_modelo.ps1
.\prediccion_en_vivo.ps1
```

After training, upload `firmware_esp32_ia.ino` to the ESP32 again using the Arduino IDE.

The serial monitor should display `clase_ia=` followed by the predicted class and a recommendation. The OLED display will show a shorter version of the recommendation.

## Using a Mobile Hotspot for the Presentation

The ESP32 only supports 2.4 GHz Wi-Fi for this project. Connect both the computer and the ESP32 to the same mobile phone hotspot.

Then, run:

```powershell
.\diagnosticar_red.ps1
```

Copy the URL displayed by the script into the `SERVER_URL` setting in the firmware. Upload the firmware to the ESP32 again, and open `prediccion_en_vivo.ps1` to test the system in real time.
