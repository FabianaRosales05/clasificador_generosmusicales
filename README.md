# Clasificador de Géneros Musicales (Redes Neuronales)

Aplicación en **Python 3.12** que clasifica una pieza musical en uno de 10 géneros (dataset GTZAN) mediante procesamiento de audio, extracción de características y redes neuronales (MLP principal + CNN como extensión). Permite **cargar un audio** o **grabar con el micrófono**, y muestra el género, la confianza, la forma de onda, el espectrograma, el Mel-espectrograma y las probabilidades por género.

```
Audio -> señal digital -> segmentos de 3 s -> características (MFCC, Chroma, ZCR, ...)
      -> vector numérico (X) -> scaler -> red neuronal (Softmax) -> género + confianza
```

> Los resultados (accuracy, F1, matriz de confusión) **no están escritos aquí a mano**: se generan al entrenar, en `resultados/`. Ver sección *Resultados*.

## Tecnologías (todas compatibles con Python 3.12)
| Librería | Uso |
|---|---|
| Librosa | Carga de audio, espectrograma, Mel, MFCC, Chroma, etc. |
| NumPy / Pandas | Vectores, matrices y tablas |
| Matplotlib | Gráficas |
| Scikit-learn | `StandardScaler`, división de datos, métricas |
| TensorFlow/Keras 2.16 | MLP y CNN |
| sounddevice + soundfile | Captura del micrófono y guardado `.wav` (únicas librerías extra; necesarias para el micrófono) |
| Tkinter | Interfaz gráfica (incluida con Python) |

## Estructura
```
clasificador_generos_musicales/
├── app.py                    Interfaz gráfica
├── config.py                 Rutas y parámetros compartidos
├── extraer_caracteristicas.py  Carga de audio, MFCC/Chroma/ZCR..., Mel (también ejecutable solo)
├── visualizacion.py          Forma de onda, espectrograma, Mel, barras de probabilidad
├── preparar_dataset.py       Construye X e y desde dataset/
├── utilidades_modelo.py      División por archivo, métricas y gráficas
├── entrenar_modelo.py        Entrena el MLP (modelo principal)
├── entrenar_cnn.py           Entrena la CNN (extensión)
├── predecir.py               Predicción de un audio nuevo (usa el mismo procesamiento)
├── grabador.py               Captura del micrófono
├── requirements.txt
├── dataset/<genero>/         Audios GTZAN (tú los colocas)
├── modelo/                   modelo_generos.keras, scaler.pkl, etiquetas.pkl (+ modelo_cnn.keras)
├── resultados/               graficas/, matrices/, predicciones/, CSV y resúmenes
└── documentacion/            Documentacion_Academica.md
```

## 1. Instalación en Windows + Visual Studio Code
1. **Instalar Python 3.12** desde python.org (Windows installer 64-bit). Marca *Add python.exe to PATH* y deja marcado *tcl/tk and IDLE* (necesario para la interfaz). Verifica: `py -3.12 --version`.
2. **Abrir la carpeta** en VS Code: *File > Open Folder* y elige `clasificador_generos_musicales`. Instala la extensión *Python* de Microsoft.
3. **Crear el entorno virtual** (terminal de VS Code, *Terminal > New Terminal*):
   ```
   py -3.12 -m venv .venv
   ```
   (equivale a `python -m venv .venv` si `python` ya es 3.12).
4. **Activarlo**:
   ```
   .venv\Scripts\activate
   ```
   Si PowerShell bloquea el script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, o usa `cmd` con `.venv\Scripts\activate.bat`.
5. **Seleccionar el intérprete**: `Ctrl+Shift+P` > *Python: Select Interpreter* > el de `.venv` (debe decir Python 3.12.x).
6. **Instalar dependencias**:
   ```
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
7. **Comprobar** versión y librerías:
   ```
   python --version
   python -c "import tensorflow, librosa, sounddevice; print('OK')"
   ```

## 2. Dataset GTZAN
Descarga GTZAN (por ejemplo la versión de Kaggle "GTZAN Dataset - Music Genre Classification", carpeta `genres_original`) y copia cada carpeta de género dentro de `dataset/`:
```
dataset\blues\blues.00000.wav ... dataset\rock\rock.00099.wav
```
Se aceptan `.wav`, `.mp3`, `.flac`, `.ogg` y `.au`. GTZAN es adecuado para fines **educativos**, pero tiene limitaciones conocidas (duplicados, errores de etiquetado y procedencia de algunos archivos); no se presenta como un benchmark científico perfecto. Algunas copias traen un archivo dañado (típicamente `jazz.00054`): el proyecto lo detecta, lo omite y lo lista en `resultados/archivos_problematicos.csv`.

## 3. Preparar datos y entrenar
```
python preparar_dataset.py      # extrae características (varios minutos)
python entrenar_modelo.py       # entrena y evalúa el MLP
python entrenar_cnn.py          # (opcional) CNN con Mel-espectrogramas
```
Opciones útiles: `python preparar_dataset.py --sin-mel` (solo MLP, más rápido); `python entrenar_modelo.py --epocas 200 --batch 64 --lr 0.0005`.

Para ver el procesamiento de un audio (evidencias 7-11):
```
python extraer_caracteristicas.py dataset\rock\rock.00000.wav
```
Crea forma de onda, espectrograma, Mel y tabla de características en `resultados/`.

## 4. Ejecutar la aplicación
```
python app.py
```
- **CARGAR AUDIO**: elige un archivo; se procesa y se muestran género, confianza y gráficas.
- **GRABAR CON MICRÓFONO**: define la duración (3-30 s, por defecto 10), pulsa el botón (la interfaz indica *GRABANDO* con contador), espera o pulsa *DETENER*. La grabación se guarda en `resultados/predicciones/` y se analiza.
- El selector **Modelo** permite elegir MLP o CNN (solo aparecen los entrenados).

Si el micrófono falla: *Configuración de Windows > Privacidad y seguridad > Micrófono* y permite el acceso a aplicaciones de escritorio; revisa que el dispositivo de entrada esté seleccionado en *Sonido*.

## 5. Probar con audio no visto
El script `entrenar_modelo.py` guarda en `resultados/division_archivos.csv` qué archivo pertenece a entrenamiento (`train`), validación (`val`) o prueba (`test`). Para la demostración:
- **Audio de prueba de GTZAN**: usa un archivo con partición `test` (el modelo nunca lo usó para ajustar pesos ni scaler).
- **Audio externo (recomendado para la sustentación)**: una canción que NO provenga de GTZAN (ej. un fragmento descargado de forma legal o grabado con micrófono).

## 6. Resultados (se generan al ejecutar)
| Archivo | Contenido |
|---|---|
| `resultados/resumen_dataset.json` | Archivos y segmentos realmente procesados |
| `resultados/caracteristicas_dataset.csv` | Tabla de características |
| `resultados/resumen_mlp.md` / `resumen_cnn.md` | Accuracy, precision, recall, F1, reporte y confusiones principales |
| `resultados/graficas/` | Pérdida y accuracy por época |
| `resultados/matrices/` | Matrices de confusión (PNG y CSV) |
| `resultados/division_archivos.csv` | Partición de cada archivo |

Estado actual del paquete: el dataset local fue procesado y el MLP entrenado. Los
artefactos conservados en `modelo/` y los resultados en `resultados/` corresponden
a esa ejecución. Para no confundirlos con una nueva corrida, este README no sustituye
las métricas generadas por los scripts: consulte `resultados/resumen_mlp.md` y
`resultados/metricas_mlp.json`.

Para la entrega final por GitHub se incluyen el informe
`documentacion/Informe_final_clasificador_generos_musicales.docx`, la
verificación del checklist y la evidencia de audio nuevo en `documentacion/`.
El profesor confirmó que no se requiere notebook; el contenido útil de su
versión anterior está documentado en
`documentacion/CONTENIDO_NOTEBOOK_RETIRADO.md`.

## 7. Decisiones de diseño
- Cada audio se divide en segmentos de 3 s (≈10 por clip de 30 s); el audio nuevo se clasifica promediando las probabilidades de sus segmentos.
- La división train/val/test se hace **por archivo** (no por segmento) para evitar que fragmentos de la misma canción aparezcan en entrenamiento y prueba.
- El `StandardScaler` se ajusta **solo con entrenamiento** y se guarda en `modelo/scaler.pkl`; la app aplica el mismo.
- El vector de entrada tiene 74 números: media y desviación estándar de 20 MFCC, 12 Chroma, ZCR, Spectral Centroid, Bandwidth, Rolloff y RMS.

## 8. Limitaciones
GTZAN es pequeño y tiene fallos conocidos; los géneros son subjetivos y se solapan; el micrófono introduce ruido de ambiente; la app analiza los primeros 60 s de un archivo; una grabación de baja calidad puede reducir la confianza.

## 9. Mejoras posibles
Aumento de datos, otros datasets (FMA), CNN más profunda o transfer learning, análisis de la sección central de la canción, mostrar las 3 mejores predicciones, empaquetar como `.exe`.

## 10. Entrega por GitHub

El repositorio está preparado para compartir código, documentación, modelo MLP,
scaler, etiquetas, tabla de características y resultados verificados. El
dataset GTZAN y la caché no se versionan porque son voluminosos o regenerables;
obtén el dataset desde una fuente autorizada si necesitas reentrenar.

Consulta estos documentos antes de publicar:

- `documentacion/ENTREGA_GITHUB.md`: pasos de publicación y reproducción.
- `documentacion/EVIDENCIA_AUDIO_NUEVO.md`: prueba real con audio externo.
- `CREDITS.md`: atribuciones, límites de licencia y uso de GTZAN.
- `LICENSE`: licencia MIT del código y documentación propios.
