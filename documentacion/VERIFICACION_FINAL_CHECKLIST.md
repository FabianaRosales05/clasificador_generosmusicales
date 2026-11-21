# Verificación final del checklist

Fecha de actualización: 20 de noviembre de 2026.

Esta verificación se elaboró sin modificar la lógica del clasificador, el modelo, el scaler, las etiquetas, el procesamiento de audio ni la interfaz.

| N.º | Requisito | Estado | Evidencia verificada |
|---:|---|---|---|
| 1 | Dataset consolidado y caracterizado | Completo | `dataset/`, `resultados/resumen_dataset.json`: 1.000 archivos encontrados, 999 procesados, 10 géneros, 9.981 segmentos y distribución por clase. |
| 2 | Carga y preprocesamiento | Completo | `extraer_caracteristicas.py` carga con Librosa, convierte a mono, usa 22.050 Hz, limita duración, valida la señal y segmenta a 3 s. |
| 3 | Visualización de señales | Completo | `resultados/guia2/graficas/` contiene waveform, espectrograma y Mel-espectrograma para blues, classical y rock; la interpretación está en `resultados/guia2/analisis_y_respuestas_guia2.md`. |
| 4 | Extracción de características | Completo | `extraer_caracteristicas.py` implementa MFCC, Chroma, ZCR, centroid, bandwidth, rolloff y RMS; el vector final tiene 74 valores. |
| 5 | Conjunto de datos y separación | Completo | `resultados/caracteristicas_dataset.csv`, `resultados/division_archivos.csv` y `entrenar_modelo.py`; la división es por archivo y el scaler se ajusta solo con train. |
| 6 | Arquitectura de red | Completo | `entrenar_modelo.py` y el informe final documentan MLP 74-128-64-10, ReLU, Dropout 0,3, Softmax, Adam y sparse categorical crossentropy. |
| 7 | Entrenamiento | Completo | `resultados/graficas/accuracy_mlp.png`, `resultados/graficas/perdida_mlp.png` y `resultados/resumen_mlp.md`; se conservaron 28 épocas realmente ejecutadas. |
| 8 | Evaluación | Completo | `resultados/metricas_mlp.json`, reportes y matrices. Por archivo: accuracy 0,7533, precision macro 0,7625, recall macro 0,7533 y F1 macro 0,7489. |
| 9 | Audio nuevo | Cumplido | `grabacion_20261014_172759.wav` no coincide con el dataset local. Se ejecutó `predecir.py` con el MLP actual: predicción `classical`, confianza 52,97 % y 3 segmentos. Evidencia en `documentacion/EVIDENCIA_AUDIO_NUEVO.md` y `resultados/predicciones/evidencia_prediccion_audio_nuevo.txt`. |
| 10 | Entrega técnica y documentación | Parcial | Están preparados README, instrucciones GitHub, `.gitignore`, `.gitattributes`, licencia, créditos, modelo, scaler, etiquetas, informe y checklist actualizado. Falta crear el repositorio remoto en la cuenta del estudiante y publicar con `git push`. |

## Verificaciones efectuadas

- Todas las dependencias declaradas se resolvieron en `.venv`: NumPy, Pandas, Librosa, scikit-learn, TensorFlow, Joblib, SoundFile, SoundDevice y Matplotlib.
- Los diez módulos Python compilaron sin errores de sintaxis.
- El modelo MLP se cargó con forma de entrada `(None, 74)` y salida `(None, 10)`; el scaler espera 74 columnas y el mapa contiene las diez etiquetas.
- La división conserva 699 archivos de entrenamiento, 150 de validación y 150 de prueba, sin archivos repetidos entre particiones.

## Adaptación autorizada para GitHub

El profesor confirmó que no se requiere notebook. El notebook anterior se
respaldó antes de retirarse y no contenía lógica funcional exclusiva; el mapeo
de su contenido está en `documentacion/CONTENIDO_NOTEBOOK_RETIRADO.md`.

El repositorio local usa la rama `main`. Antes de publicar, siga
`documentacion/ENTREGA_GITHUB.md` y compruebe que `dataset/`, `cache/`,
`.venv/` y la carpeta de respaldo no se agreguen a Git.

## Limitación declarada

GTZAN tiene limitaciones conocidas de duplicados, procedencia y etiquetas. El archivo `jazz/jazz.00054.wav` no pudo procesarse y está registrado en `resultados/archivos_problematicos.csv`; no se empleó en el entrenamiento ni en la evaluación.
