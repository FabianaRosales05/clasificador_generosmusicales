# Clasificador de Géneros Musicales mediante Procesamiento de Audio y Redes Neuronales

**Asignatura:** Redes Neuronales · **Entorno:** Python 3.12, Visual Studio Code, Windows
**Estudiante(s):** _[completar]_ · **Docente:** _[completar]_ · **Fecha:** _[completar]_

> **Nota de honestidad académica.** Todas las cifras de la sección *Resultados* deben provenir de ejecutar el proyecto con el dataset real. Este documento NO contiene métricas inventadas; donde falta una ejecución, se marca **PENDIENTE**.

---

## 1. Introducción
Una persona reconoce con facilidad si una canción "suena a rock" o a música clásica; un computador, en cambio, solo recibe números. Este proyecto construye un sistema que convierte el audio en información numérica (características) y la usa como entrada de una red neuronal que predice el género musical. El sistema se entrega como una aplicación de escritorio que permite cargar un archivo o grabar con el micrófono.

## 2. Planteamiento del problema
Identificar automáticamente el género de una pieza a partir de su señal de audio es difícil porque: (a) la señal cruda es una larga secuencia de amplitudes que no expone directamente el timbre ni la armonía; (b) los géneros son categorías culturales y subjetivas, con fronteras difusas (por ejemplo rock/metal, o pop/disco); (c) piezas del mismo género varían mucho en instrumentación y producción; (d) las grabaciones reales incluyen ruido. Se requiere, por tanto, una representación numérica informativa y un modelo capaz de aprender patrones a partir de ejemplos.

## 3. Justificación
El procesamiento de audio (espectrogramas, MFCC) resume la información relevante de la señal en pocas variables; las redes neuronales aprenden relaciones no lineales entre esas variables y la clase sin reglas escritas a mano. El problema es un caso claro de clasificación multiclase (10 clases), útil para organizar bibliotecas musicales y sistemas de recomendación, y didáctico para comprender el flujo completo *datos → características → modelo → evaluación → aplicación*.

## 4. Objetivo general
Desarrollar una aplicación que clasifique automáticamente una pieza musical en uno de 10 géneros del dataset GTZAN mediante extracción de características de audio y redes neuronales, a partir de un archivo cargado o de una grabación por micrófono.

## 5. Objetivos específicos
1. **Procesamiento de audio:** cargar audio con Librosa y visualizar forma de onda, espectrograma y Mel-espectrograma.
2. **Extracción de características:** calcular ZCR, Spectral Centroid, Bandwidth, Rolloff, Chroma y MFCC.
3. **Construcción del dataset:** organizar `X` (características) e `y` (género) sin perder la correspondencia audio–etiqueta, detectando archivos problemáticos.
4. **Entrenamiento:** entrenar un MLP (Keras) con normalización sin fuga de información y control de sobreajuste.
5. **Evaluación:** medir accuracy, precision, recall, F1, matriz de confusión y analizar confusiones entre géneros.
6. **Aplicación gráfica:** construir una interfaz minimalista que muestre género, confianza y visualizaciones.
7. **Predicción por archivo:** procesar un audio nuevo con el mismo pipeline del entrenamiento.
8. **Predicción por micrófono:** capturar, guardar, procesar y clasificar una grabación.
9. **Extensión:** entrenar una CNN sobre Mel-espectrogramas y compararla conceptualmente con el MLP.

## 6. Marco conceptual
**Audio digital y señal de audio.** El sonido es una variación de presión; al digitalizarlo se mide su amplitud a intervalos regulares, obteniendo una secuencia de números (`audio` en el código).
**Frecuencia de muestreo (`sr`).** Número de muestras por segundo (aquí 22 050 Hz). Duración = `len(audio)/sr`.
**Forma de onda.** Gráfica de la amplitud en función del tiempo.
**Frecuencia.** Número de oscilaciones por segundo (Hz); relacionada con lo grave o agudo de un sonido.
**FFT.** Algoritmo que descompone una señal en las frecuencias que la componen. Aplicada por ventanas cortas de tiempo (STFT) da el espectrograma.
**Espectrograma.** Eje horizontal = tiempo; eje vertical = frecuencia; color = energía/intensidad en esa frecuencia y ese instante.
**Mel-espectrograma.** Espectrograma cuyo eje de frecuencia usa la escala Mel, aproximación a la percepción auditiva humana (más resolución en graves). Es una "imagen" adecuada como entrada de una CNN.
**Características de audio.** Medidas numéricas del sonido; **no son el género**, sino la información que el modelo usa para decidirlo.
- **ZCR (Zero Crossing Rate):** proporción de veces que la señal cruza el cero; mayor en sonidos ruidosos o con mucho agudo.
- **Spectral Centroid:** "centro de masa" del espectro; indica si el sonido es brillante (alto) u oscuro (bajo).
- **Spectral Bandwidth:** dispersión de las frecuencias alrededor del centroide.
- **Spectral Rolloff:** frecuencia bajo la cual se concentra el 85 % de la energía espectral.
- **Chroma:** energía en las 12 clases de tono (Do…Si); describe la armonía.
- **MFCC:** coeficientes cepstrales en escala Mel; resumen compacto del timbre. **No son el género**: son números de entrada para el modelo.
**Red neuronal.** Conjunto de neuronas organizadas en capas. **Neurona:** calcula una suma ponderada de entradas más un sesgo y aplica una activación. **Capa Dense:** cada neurona se conecta con todas las de la capa anterior. **ReLU:** `f(x)=max(0,x)`; aporta no linealidad. **Softmax:** convierte las salidas en probabilidades que suman 1.
**Función de pérdida.** Mide el error del modelo; se usa *sparse categorical crossentropy* (clases como enteros). **Optimizador:** Adam, ajusta los pesos para reducir la pérdida.
**Época:** una pasada completa por los datos de entrenamiento. **Batch size:** muestras procesadas antes de actualizar los pesos. **Learning rate:** tamaño del paso de actualización.
**Overfitting:** el modelo memoriza el entrenamiento y generaliza mal; se controla con Dropout, EarlyStopping y evaluación en datos no vistos.
**Métricas.** *Accuracy:* proporción de aciertos. *Precision:* de lo que predijo como un género, cuánto era correcto. *Recall:* de lo que realmente era ese género, cuánto detectó. *F1:* media armónica de precision y recall. *Matriz de confusión:* tabla real vs. predicho; la diagonal son aciertos y fuera de ella las confusiones.

## 7. Dataset
Se usa **GTZAN Genre Collection**: según su descripción original, 10 géneros (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock) con clips de aproximadamente 30 s. **Limitaciones conocidas:** presencia de duplicados, errores de etiquetado, procedencia irregular de algunos archivos y distorsiones (Sturm, 2013). Por eso se emplea con **fines educativos** y no como benchmark científico perfecto. La cantidad exacta de archivos procesados es la que reporta `resultados/resumen_dataset.json` tras `preparar_dataset.py` (**PENDIENTE de ejecutar**).

## 8. Metodología
```
Audio → Preprocesamiento → Extracción de características → Representación numérica
      → Preparación de datos → Entrenamiento → Evaluación → Predicción
```
1. **Preprocesamiento** (`extraer_caracteristicas.cargar_audio`): lectura con Librosa a 22 050 Hz, mono; validación (vacío, silencio, muy corto, valores inválidos). El audio se divide en **segmentos de 3 s**, lo que multiplica los ejemplos por ~10 y permite clasificar grabaciones cortas.
2. **Características** (`extraer_vector`): por segmento se calculan MFCC (20), Chroma (12), ZCR, Centroid, Bandwidth, Rolloff y RMS (energía).
3. **Representación numérica:** cada característica se calcula por ventanas de tiempo; para obtener un **vector de tamaño fijo** se resume con **media y desviación estándar** de cada fila. Resultado: **74 números por segmento** (2×(20+12+1+1+1+1+1)). Se construye `X` (segmentos × 74) e `y` (género), guardados en la misma fila del CSV, de modo que no se pierde la correspondencia.
4. **Preparación:** limpieza de NaN/infinitos; archivos ilegibles se omiten y se registran. **División por archivo** estratificada: 70 % entrenamiento, 15 % validación, 15 % prueba (todos los segmentos de una canción quedan en una sola partición para evitar fuga de información). `StandardScaler` ajustado **solo con entrenamiento**, luego aplicado a validación, prueba y audios nuevos; se guarda en `scaler.pkl`.
5. **Entrenamiento** (`entrenar_modelo.py`): Adam (lr 0,001), batch 32, hasta 150 épocas con EarlyStopping (paciencia 15, restaura mejores pesos) y reducción del learning rate en meseta.
6. **Evaluación:** sobre la partición de prueba, por segmento y por archivo (promediando probabilidades de los segmentos).
7. **Predicción:** un audio nuevo pasa por el mismo pipeline; las probabilidades de sus segmentos se promedian; el género es la clase de mayor probabilidad y la **confianza** es esa probabilidad (en %).

## 9. Arquitectura del modelo (MLP)
```
Entrada (74) → Dense(128, ReLU) → Dropout(0,3) → Dense(64, ReLU) → Dropout(0,3) → Dense(10, Softmax)
```
Se eligieron 128 y 64 neuronas porque la entrada es de tamaño moderado (74): suficiente capacidad para aprender combinaciones de características sin un modelo grande que se sobreajuste con pocos datos y que sea fácil de explicar. La salida tiene 10 neuronas (una por género) con Softmax para obtener probabilidades. Dropout reduce el sobreajuste. El número de parámetros y la tabla de capas se imprimen en consola al entrenar (`modelo.summary()`).

## 10. CNN (extensión)
```
Audio → segmento 3 s → Mel-espectrograma (64×130) → [Conv2D+ReLU+BatchNorm+MaxPool]×3 → Flatten → Dense(64) → Softmax(10)
```
El MLP recibe solo resúmenes estadísticos (promedios y desviaciones), por lo que pierde el orden temporal y la forma de los patrones. La CNN recibe el Mel-espectrograma completo como una imagen; sus filtros detectan patrones locales tiempo-frecuencia (armónicos, ataques de percusión) y las capas siguientes los combinan. Se usa el mismo procedimiento de partición por archivo para poder comparar ambos modelos. **No se afirma que la CNN sea mejor**: eso debe verificarse con `resumen_mlp.md` frente a `resumen_cnn.md`.

## 11. Desarrollo de la aplicación (`app.py`)
Interfaz Tkinter de tema oscuro con título **CLASIFICADOR DE GÉNEROS MUSICALES**, dos botones principales (**CARGAR AUDIO** y **GRABAR CON MICRÓFONO**), selector de modelo (MLP/CNN), duración de grabación configurable, barra de estado, tarjeta con *GÉNERO DETECTADO* y *Confianza*, y cuatro gráficas calculadas para el audio analizado: forma de onda, espectrograma, Mel-espectrograma y probabilidades de los 10 géneros. El análisis se ejecuta en un hilo aparte para no congelar la ventana. Errores (archivo no compatible, dañado, silencio, modelo ausente, micrófono) se muestran en cuadros de diálogo con mensajes comprensibles.

## 12. Grabación mediante micrófono
```
Pulsar GRABAR → abrir micrófono (sounddevice) → captura en bloques → detener (manual o al cumplir la duración)
→ guardar .wav (soundfile) en resultados/predicciones → cargar y procesar con Librosa → características
→ scaler → red neuronal → género y confianza → gráficas
```
`grabador.py` captura a 22 050 Hz si el dispositivo lo admite (si no, a su frecuencia nativa y Librosa remuestrea). Maneja: librería no disponible, ausencia de dispositivo de entrada, falta de permisos o dispositivo ocupado, grabación vacía, error al guardar y silencio.

## 13. Pruebas
Registrar aquí las pruebas **realmente ejecutadas** (fecha, entrada, resultado esperado y obtenido, evidencia).
| # | Prueba | Resultado esperado | Resultado obtenido |
|---|---|---|---|
| 1 | Extracción de características de un audio de GTZAN | Vector de 74 números, gráficas guardadas | PENDIENTE |
| 2 | Preparación del dataset completo | CSV y `resumen_dataset.json` | PENDIENTE |
| 3 | Entrenamiento del MLP | Modelo, scaler y etiquetas guardados | PENDIENTE |
| 4 | Entrenamiento de la CNN | `modelo_cnn.keras` | PENDIENTE |
| 5 | Carga de audio de la partición `test` | Género y confianza | PENDIENTE |
| 6 | Carga de audio externo (no GTZAN) | Género y confianza | PENDIENTE |
| 7 | Grabación con micrófono | Archivo `.wav` y predicción | PENDIENTE |
| 8 | Archivo con formato no compatible / dañado | Mensaje de error, sin cierre de la app | PENDIENTE |
| 9 | Audio en silencio / grabación vacía | Mensaje de error | PENDIENTE |
| 10 | Micrófono desconectado | Mensaje de error | PENDIENTE |

*Verificación técnica realizada durante el desarrollo del código:* el flujo (preparación → entrenamiento MLP y CNN → predicción → manejo de errores → interfaz) se ejecutó con un conjunto de **audios sintéticos generados artificialmente** solo para comprobar que el código funciona. Esos números no son resultados del proyecto y no deben citarse. La captura por micrófono no pudo probarse en ese entorno (sin hardware de audio) y debe probarse en el computador del estudiante.

## 14. Resultados
**PENDIENTE:** completar tras ejecutar con GTZAN. Fuentes reales:
- Conteos: `resultados/resumen_dataset.json`.
- Métricas MLP: `resultados/resumen_mlp.md`, `reporte_mlp_archivos.txt`.
- Métricas CNN: `resultados/resumen_cnn.md`.
- Gráficas: `resultados/graficas/perdida_mlp.png`, `accuracy_mlp.png` (y `_cnn`).
- Matrices: `resultados/matrices/matriz_confusion_mlp_archivos.png`.

Plantilla (pegar valores reales): Accuracy por archivo = ___ ; F1 macro = ___ ; géneros mejor clasificados = ___ ; géneros más confundidos = ___ (ver "Confusiones más frecuentes" en el resumen).

## 15. Análisis
**PENDIENTE** (depende de los resultados). Guía: (1) comparar las curvas de entrenamiento y validación: si la pérdida de validación sube mientras la de entrenamiento baja, hay sobreajuste; (2) identificar en la matriz de confusión los pares de géneros más confundidos y relacionarlos con características compartidas (p. ej., rock/metal comparten guitarras distorsionadas; pop/disco comparten ritmo y producción); (3) comparar accuracy por segmento y por archivo; (4) comparar MLP vs CNN; (5) contrastar la prueba con audio externo.

## 16. Limitaciones
- GTZAN: tamaño reducido, duplicados y errores de etiquetado; los resultados no se extrapolan sin cautela.
- Los géneros son subjetivos y se solapan; la "confianza" es la probabilidad de Softmax, no una garantía de acierto.
- Audio externo con producción, ruido o idioma distintos puede degradar el desempeño.
- La app analiza los primeros 60 s de un archivo.
- Las grabaciones por micrófono incluyen ruido ambiental y respuesta del equipo.
- El MLP pierde información temporal al resumir con media y desviación.

## 17. Conclusiones
**PENDIENTE de redactar con los resultados reales.** Aspectos que sí pueden afirmarse del trabajo ya construido: se implementó un pipeline completo y reproducible audio → características → red neuronal → aplicación; se evitó fuga de información (scaler ajustado solo con entrenamiento; división por archivo); y se incluyó una extensión CNN. Las afirmaciones sobre desempeño deben basarse en los archivos de `resultados/`.

## 18. Trabajo futuro
Aumento de datos (ruido, cambio de tono/velocidad), validación cruzada, CNN más profunda o transfer learning, uso de otros datasets (p. ej. FMA), análisis de la parte central de la canción, mostrar las tres mejores predicciones, calibración de la confianza y empaquetado como ejecutable.

## 19. Referencias
- Material del curso: *Guía 2 – Procesamiento y extracción de características del audio* y documento *Lenguaje de desarrollo* (Redes Neuronales, 6.º semestre).
- Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. *IEEE Transactions on Speech and Audio Processing, 10*(5), 293–302.
- Sturm, B. L. (2013). The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use. arXiv:1306.1461.
- McFee, B., et al. (2015). librosa: Audio and music signal analysis in Python. *Proceedings of the 14th Python in Science Conference.*
- Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition. *IEEE Trans. ASSP, 28*(4).
- Abadi, M., et al. (2016). TensorFlow: A system for large-scale machine learning. *OSDI.* · Chollet, F. (2015). Keras. https://keras.io
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12.*
- Documentación oficial de Librosa, TensorFlow/Keras, Scikit-learn y sounddevice.

---

## Anexo A. Evidencias que el proyecto permite obtener
| # | Evidencia | Cómo obtenerla |
|---|---|---|
| 1-2 | Entorno y Python 3.12 | VS Code; `python --version` |
| 3-4 | Dependencias y estructura | `pip install -r requirements.txt`; explorador de VS Code |
| 5 | Dataset | carpeta `dataset/` |
| 6 | Carga de audio | `python extraer_caracteristicas.py <audio>` (muestra sr, muestras, duración) |
| 7-9 | Forma de onda, espectrograma, Mel | `resultados/graficas/*_<audio>.png` o la app |
| 10-11 | Características y tabla | consola y `resultados/tabla_caracteristicas_*.csv`; `caracteristicas_dataset.csv` |
| 12-15 | Entrenamiento, loss/accuracy, matriz, métricas | consola de `entrenar_modelo.py`; `resultados/graficas`, `matrices`, `resumen_mlp.md` |
| 16-20 | Interfaz, audio nuevo, micrófono, predicción, género y confianza | `python app.py` |

## Anexo B. Prueba con audio no visto
- **Entrenamiento:** ajusta los pesos y el scaler. **Validación:** vigila el sobreajuste y decide cuándo detener. **Prueba:** se usa una sola vez al final para medir (`division_archivos.csv`, partición `test`).
- **Audio externo/no visto:** canción ajena a GTZAN o grabación propia; es la demostración más sólida. Procedimiento: abrir `app.py` → *CARGAR AUDIO* (o grabar) → anotar género, confianza y capturar las gráficas. Aclarar en la exposición que la predicción de un solo audio es una demostración, no una métrica.

## Anexo C. Preguntas y respuestas para la sustentación
1. **¿Qué problema resuelve?** Clasifica automáticamente una pieza musical en un género a partir de su audio.
2. **¿Qué dataset utilizó?** GTZAN Genre Collection, con fines educativos.
3. **¿Cuántos géneros maneja?** Diez: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae y rock.
4. **¿Qué es una señal de audio?** Una secuencia de números que mide la amplitud del sonido en instantes regulares.
5. **¿Qué es un espectrograma?** Una gráfica de tiempo (horizontal), frecuencia (vertical) y energía (color) que muestra cómo cambia el contenido de frecuencias.
6. **¿Qué es un Mel-espectrograma?** Un espectrograma con la frecuencia en escala Mel, más parecida a como oímos.
7. **¿Qué es MFCC?** Coeficientes numéricos que resumen el timbre del sonido a partir del espectro en escala Mel. No son el género.
8. **¿Por qué se utilizan MFCC?** Porque resumen en pocos números la información de timbre, que ayuda a distinguir géneros e instrumentos.
9. **¿Qué es ZCR?** La frecuencia con que la señal cambia de signo; es alta en sonidos ruidosos o muy agudos.
10. **¿Qué es Spectral Centroid?** El "centro de masa" del espectro: indica si el sonido es más brillante o más grave.
11. **¿Qué es Spectral Bandwidth?** Qué tan dispersas están las frecuencias alrededor del centroide.
12. **¿Qué es Spectral Rolloff?** La frecuencia bajo la cual está el 85 % de la energía del espectro.
13. **¿Qué es Chroma?** La energía en cada una de las 12 notas musicales; describe la armonía.
14. **¿Por qué convertir el audio en números?** Porque la red neuronal solo opera con números; las características son la información útil extraída del audio.
15. **¿Qué es una neurona?** Una unidad que suma sus entradas multiplicadas por pesos, añade un sesgo y aplica una activación.
16. **¿Qué es una capa Dense?** Una capa donde cada neurona está conectada con todas las de la capa anterior.
17. **¿Qué hace ReLU?** Deja pasar los valores positivos y convierte los negativos en cero, aportando no linealidad.
18. **¿Qué hace Softmax?** Convierte las 10 salidas en probabilidades que suman 1; la mayor es el género predicho.
19. **¿Qué es una época?** Una pasada completa por todos los datos de entrenamiento.
20. **¿Qué es batch size?** El número de muestras que se procesan antes de actualizar los pesos (aquí 32).
21. **¿Qué es learning rate?** El tamaño del paso con que se corrigen los pesos (aquí 0,001 inicial).
22. **¿Qué es overfitting?** Cuando el modelo memoriza el entrenamiento y falla con datos nuevos; se reduce con Dropout y EarlyStopping.
23. **¿Qué es accuracy?** El porcentaje de predicciones correctas sobre el total.
24. **¿Qué es precision?** De todo lo que el modelo llamó "rock", qué proporción realmente era rock.
25. **¿Qué es recall?** De todo lo que realmente era rock, qué proporción detectó el modelo.
26. **¿Qué es F1-score?** La media armónica de precision y recall; resume ambos en un número.
27. **¿Qué representa la matriz de confusión?** Filas = género real, columnas = predicho; la diagonal son aciertos y el resto, confusiones.
28. **¿Por qué algunos géneros pueden confundirse?** Porque comparten instrumentos, ritmos o producción (por ejemplo rock y metal), y los géneros son subjetivos. Qué pares se confunden en este proyecto lo indica la matriz real.
29. **¿Cómo funciona la predicción de un audio nuevo?** Se carga, se divide en segmentos de 3 s, se extraen las 74 características, se aplica el scaler guardado, la red calcula probabilidades, se promedian y se elige la mayor.
30. **¿Cómo funciona la grabación por micrófono?** Se captura con sounddevice, se guarda como .wav y se procesa igual que un archivo cargado.
31. **¿Por qué se utiliza un scaler?** Porque las características tienen escalas muy distintas; normalizarlas ayuda a que la red aprenda bien.
32. **¿Por qué el scaler se ajusta solo con entrenamiento?** Para que validación y prueba no influyan en el procesamiento; si no, habría fuga de información y las métricas serían optimistas.
33. **¿Diferencia entre MLP y CNN?** El MLP recibe un vector de características; la CNN recibe el Mel-espectrograma como imagen y detecta patrones locales.
34. **¿Por qué Mel-espectrogramas para una CNN?** Porque son imágenes tiempo-frecuencia donde los patrones musicales tienen forma, algo que las CNN procesan bien.
35. **¿Cuáles son las limitaciones?** Dataset pequeño con fallos conocidos, géneros subjetivos, ruido en grabaciones y desempeño que solo se confirma con los resultados reales obtenidos.

## Anexo D. Revisión de compatibilidad con Python 3.12
- TensorFlow 2.16.2 soporta Python 3.12 y exige NumPy < 2.0 → se fija NumPy 1.26.4.
- Pandas 2.2.2, Matplotlib 3.9.2, Librosa 0.10.2.post1, Scikit-learn 1.5.2, soundfile 0.12.1 y joblib 1.4.2 son compatibles con Python 3.12 y NumPy 1.26.
- sounddevice 0.5.1 incluye PortAudio en Windows. Tkinter viene con el instalador oficial de Python 3.12.
- No se usa Python 3.11 en ninguna parte.
