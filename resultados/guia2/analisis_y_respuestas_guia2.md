# Guía 2 — análisis de 15 audios

## Muestra procesada

Se procesaron cinco archivos WAV de cada género: blues, classical y rock. Cada
fila de `tabla_consolidada_15_audios.csv` representa una canción. Para mantener
coherencia con el clasificador, sus vectores de segmentos de 3 s se promediaron
por archivo. La tabla contiene información básica y 74 características: medias
y desviaciones de MFCC, Chroma, ZCR, centroide, ancho de banda, rolloff y RMS.

## Comparación observada en esta muestra

Los promedios siguientes proceden exclusivamente de los 15 archivos procesados:

- **zcr_media**: mínimo en `rock` (0.06069999933242798) y máximo en `classical` (0.09860000014305115).
- **centroide_media**: mínimo en `classical` (1520.594482421875) y máximo en `rock` (1634.452880859375).
- **ancho_banda_media**: mínimo en `classical` (1554.4892578125) y máximo en `rock` (1938.4056396484375).
- **rolloff_media**: mínimo en `classical` (2799.9755859375) y máximo en `rock` (3483.250732421875).
- **rms_media**: mínimo en `classical` (0.03629999980330467) y máximo en `rock` (0.16259999573230743).
- **mfcc1_media**: mínimo en `classical` (-285.2850036621094) y máximo en `rock` (-127.05390167236328).
- **mfcc2_media**: mínimo en `rock` (125.7771987915039) y máximo en `classical` (135.08389282226562).
- **chroma1_media**: mínimo en `classical` (0.2313999980688095) y máximo en `rock` (0.38420000672340393).

Estas diferencias describen esta selección, no una regla universal de cada
género. Ninguna característica aislada identifica por sí sola un género: los
valores pueden solaparse entre canciones. Por ello el MLP recibe el vector
completo de 74 características y aprende combinaciones de patrones.

## Respuestas de cierre

1. El audio se transforma en números porque una red neuronal opera sobre
   entradas numéricas de forma y escala definidas.
2. La forma de onda muestra amplitud frente al tiempo; el espectrograma añade
   frecuencia y energía a lo largo del tiempo.
3. El análisis frecuencial revela qué bandas están presentes y cómo cambia su
   energía, información que la onda sola no separa claramente.
4. Una característica es una medida numérica extraída de la señal que resume un
   aspecto útil para una tarea de aprendizaje.
5. Los MFCC resumen la envolvente espectral en escala Mel; describen timbre, no
   el género directamente.
6. Una sola medida pierde armonía, timbre, energía y variación temporal; varias
   características reducen esa pérdida de información.
7. Las características forman las columnas de entrada X; el género de la
   carpeta es la etiqueta y.
8. No. Canciones del mismo género pueden diferir en instrumentos, mezcla,
   intérprete y sección musical.
9. Un desbalance daría más peso a Rock y puede perjudicar el reconocimiento de
   Jazz; por eso deben revisarse conteos y usarse división estratificada.
10. Se necesitan audios legibles, etiquetas correctas, una tabla de
    características sin valores inválidos y una división train/validación/prueba
    sin fuga por archivo.
