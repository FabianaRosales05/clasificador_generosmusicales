"""Genera las evidencias reproducibles de la Guía 2 con 15 audios de GTZAN.

Reutiliza el mismo preprocesamiento y las mismas visualizaciones del proyecto;
no prepara ni entrena el modelo. Por defecto usa cinco archivos de blues,
classical y rock. Ejemplo:

    .\\.venv\\Scripts\\python.exe generar_evidencias_guia2.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config
import extraer_caracteristicas as ec
import visualizacion as vz


GENEROS_GUIA = ("blues", "classical", "rock")


def seleccionar_archivos(generos, cantidad):
    """Selecciona de manera determinista los primeros WAV por género."""
    seleccion = []
    for genero in generos:
        carpeta = config.DIR_DATASET / genero
        archivos = sorted(p for p in carpeta.glob("*.wav") if p.is_file())
        if len(archivos) < cantidad:
            raise ValueError(f"{genero} tiene {len(archivos)} WAV; se requieren {cantidad}.")
        seleccion.extend((genero, ruta) for ruta in archivos[:cantidad])
    return seleccion


def fila_archivo(genero, ruta):
    """Obtiene una fila por canción, promediando sus vectores de 3 segundos."""
    audio, sr = ec.cargar_audio(ruta, duracion_max=config.DURACION_MAX_ENTRENAMIENTO)
    segmentos = ec.dividir_segmentos(audio)
    vectores = np.stack([ec.extraer_vector(s) for s in segmentos])
    valores = vectores.mean(axis=0)
    fila = {
        "archivo": ruta.name,
        "genero": genero,
        "frecuencia_muestreo_hz": sr,
        "muestras": len(audio),
        "duracion_s": round(len(audio) / sr, 3),
        "segmentos_3s": len(segmentos),
    }
    fila.update(dict(zip(ec.nombres_caracteristicas(), valores)))
    return fila, audio, sr


def guardar_figuras(evidencia, ejemplos):
    carpeta = evidencia / "graficas"
    carpeta.mkdir(parents=True, exist_ok=True)
    for genero, (audio, sr) in ejemplos.items():
        for nombre, figura in (
            ("forma_onda", vz.fig_forma_onda(audio, sr, oscuro=False)),
            ("espectrograma", vz.fig_espectrograma(audio, sr, oscuro=False)),
            ("mel_espectrograma", vz.fig_mel(audio, sr, oscuro=False)),
        ):
            figura.savefig(carpeta / f"{nombre}_{genero}.png", dpi=150)


def escribir_analisis(tabla, evidencia):
    columnas = ["zcr_media", "centroide_media", "ancho_banda_media", "rolloff_media",
               "rms_media", "mfcc1_media", "mfcc2_media", "chroma1_media"]
    resumen = tabla.groupby("genero")[columnas].mean().round(4)
    resumen.to_csv(evidencia / "promedios_por_genero.csv")

    diferencias = []
    for columna in columnas:
        minimo = resumen[columna].idxmin()
        maximo = resumen[columna].idxmax()
        diferencias.append(
            f"- **{columna}**: mínimo en `{minimo}` ({resumen.loc[minimo, columna]}) "
            f"y máximo en `{maximo}` ({resumen.loc[maximo, columna]}).")

    texto = """# Guía 2 — análisis de 15 audios

## Muestra procesada

Se procesaron cinco archivos WAV de cada género: blues, classical y rock. Cada
fila de `tabla_consolidada_15_audios.csv` representa una canción. Para mantener
coherencia con el clasificador, sus vectores de segmentos de 3 s se promediaron
por archivo. La tabla contiene información básica y 74 características: medias
y desviaciones de MFCC, Chroma, ZCR, centroide, ancho de banda, rolloff y RMS.

## Comparación observada en esta muestra

Los promedios siguientes proceden exclusivamente de los 15 archivos procesados:

""" + "\n".join(diferencias) + """

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
"""
    (evidencia / "analisis_y_respuestas_guia2.md").write_text(texto, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Genera evidencias de la Guía 2.")
    parser.add_argument("--cantidad", type=int, default=5, help="Audios por género.")
    parser.add_argument("--generos", nargs=3, default=GENEROS_GUIA,
                        help="Exactamente tres géneros del dataset.")
    parser.add_argument("--reproducir", action="store_true",
                        help="Reproduce el primer audio seleccionado al finalizar.")
    args = parser.parse_args()
    if args.cantidad < 1:
        parser.error("--cantidad debe ser al menos 1.")

    config.crear_carpetas()
    evidencia = config.DIR_RESULTADOS / "guia2"
    evidencia.mkdir(parents=True, exist_ok=True)
    filas, ejemplos, problemas = [], {}, []
    for genero, ruta in seleccionar_archivos(args.generos, args.cantidad):
        try:
            fila, audio, sr = fila_archivo(genero, ruta)
            filas.append(fila)
            ejemplos.setdefault(genero, (audio, sr))
            print(f"OK {genero}/{ruta.name}")
        except ec.ErrorAudio as error:
            problemas.append({"archivo": str(ruta), "genero": genero, "error": str(error)})
            print(f"ERROR {genero}/{ruta.name}: {error}")

    if problemas:
        pd.DataFrame(problemas).to_csv(evidencia / "archivos_problematicos.csv", index=False)
    if not filas:
        raise SystemExit("No se pudo procesar ningún audio.")

    tabla = pd.DataFrame(filas)
    tabla.to_csv(evidencia / "tabla_consolidada_15_audios.csv", index=False)
    guardar_figuras(evidencia, ejemplos)
    escribir_analisis(tabla, evidencia)
    print(f"\nEvidencias creadas en: {evidencia}")
    print(f"Audios correctos: {len(tabla)} | problemáticos: {len(problemas)}")
    print(f"Tabla: {evidencia / 'tabla_consolidada_15_audios.csv'}")
    if args.reproducir:
        try:
            import sounddevice as sd
            _, ruta = seleccionar_archivos(args.generos, args.cantidad)[0]
            audio, sr = ec.cargar_audio(ruta, duracion_max=config.DURACION_MAX_ENTRENAMIENTO)
            print(f"Reproduciendo: {ruta.name}")
            sd.play(audio, sr)
            sd.wait()
        except Exception as error:
            print(f"No se pudo reproducir el audio: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
