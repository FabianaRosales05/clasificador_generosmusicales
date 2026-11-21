"""
extraer_caracteristicas.py
==========================
Convierte un audio en NÚMEROS que una red neuronal pueda usar.

    Archivo de audio -> señal digital -> segmentos de 3 s
        -> características (MFCC, Chroma, ZCR, ...) -> vector numérico

Este módulo lo usan tres partes del proyecto:
  * preparar_dataset.py  (construye X para entrenar)
  * predecir.py          (procesa un audio NUEVO exactamente igual)
  * app.py               (a través de predecir.py)

También se puede ejecutar solo para ver el procesamiento de un audio:

    python extraer_caracteristicas.py ruta\\al\\audio.wav
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

import config


class ErrorAudio(Exception):
    """Problema con el archivo o la señal de audio (mensaje apto para el usuario)."""


# =====================================================================
# 1. CARGA Y VALIDACIÓN DEL AUDIO
# =====================================================================
def cargar_audio(ruta, duracion_max=None):
    """
    Carga un archivo de audio y devuelve (audio, sr).

    audio : arreglo NumPy 1D. Cada elemento es la AMPLITUD de la señal en un
            instante (valores entre -1 y 1). Es la "forma de onda" en números.
    sr    : frecuencia de muestreo = muestras por segundo (aquí 22050).
            Tiempo de cada muestra = 1 / sr segundos.
            Duración del audio     = len(audio) / sr segundos.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise ErrorAudio(f"El archivo no existe:\n{ruta}")
    if ruta.suffix.lower() not in config.EXTENSIONES:
        raise ErrorAudio(
            f"Formato '{ruta.suffix}' no compatible.\n"
            f"Formatos permitidos: {', '.join(config.EXTENSIONES)}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # mono=True -> mezcla los canales en uno solo
            # sr=config.SR -> remuestrea para que TODOS los audios tengan igual sr
            audio, sr = librosa.load(ruta, sr=config.SR, mono=True,
                                     duration=duracion_max)
    except Exception as e:
        raise ErrorAudio("No se pudo leer el audio (¿archivo dañado o formato no soportado?)."
                         f"\nDetalle: {type(e).__name__} {e}") from e
    validar_audio(audio)
    return audio, sr


def validar_audio(audio):
    """Detecta audios vacíos, con valores inválidos, muy cortos o en silencio."""
    if audio is None or len(audio) == 0:
        raise ErrorAudio("El audio está vacío (no contiene muestras).")
    if not np.all(np.isfinite(audio)):
        raise ErrorAudio("El audio contiene valores inválidos (NaN/infinito).")
    if len(audio) / config.SR < config.DURACION_MIN_SEG:
        raise ErrorAudio(f"El audio es demasiado corto "
                         f"(mínimo {config.DURACION_MIN_SEG:.0f} segundo).")
    if np.max(np.abs(audio)) < config.UMBRAL_SILENCIO:
        raise ErrorAudio("El audio es silencio: no se detectó sonido.\n"
                         "Si grabaste con micrófono, revisa que esté activo.")


def dividir_segmentos(audio):
    """
    Divide el audio en ventanas de SEGMENTO_SEG segundos (3 s por defecto).

    Por qué: (1) un clip de 30 s da ~10 ejemplos de entrenamiento en vez de 1;
    (2) la grabación del micrófono puede ser corta. Si el audio es más corto
    que una ventana, se usa completo como un único segmento.
    """
    n = int(config.SEGMENTO_SEG * config.SR)
    if len(audio) < n:
        return [audio]
    return [audio[i * n:(i + 1) * n] for i in range(len(audio) // n)]


# =====================================================================
# 2. CARACTERÍSTICAS -> VECTOR NUMÉRICO (entrada del MLP)
# =====================================================================
def _estadisticos(matriz):
    """
    Cada característica se calcula por ventanas de tiempo (una matriz
    [coeficientes x tiempo]). Para obtener un vector de TAMAÑO FIJO se resume
    el tiempo con el PROMEDIO y la DESVIACIÓN ESTÁNDAR de cada fila.
    """
    matriz = np.atleast_2d(matriz)
    return np.concatenate([matriz.mean(axis=1), matriz.std(axis=1)])


def extraer_vector(segmento):
    """
    Devuelve el vector de características (74 números) de un segmento.

    IMPORTANTE: estas características NO son el género. Son medidas numéricas
    del sonido; la red neuronal aprende a relacionarlas con el género.
    """
    kw = dict(n_fft=config.N_FFT, hop_length=config.HOP)
    sr = config.SR

    # MFCC: resumen del "timbre". Describe la forma de la envolvente espectral
    # en escala Mel (escala cercana a la percepción auditiva humana).
    mfcc = librosa.feature.mfcc(y=segmento, sr=sr, n_mfcc=config.N_MFCC, **kw)

    # Chroma: energía en cada una de las 12 notas (Do, Do#, Re, ...). Informa
    # sobre la armonía/tonalidad.
    chroma = librosa.feature.chroma_stft(y=segmento, sr=sr, **kw)

    # ZCR (Zero Crossing Rate): con qué frecuencia la señal cruza el cero.
    # Valores altos -> sonidos ruidosos/agudos (p. ej. metal, platillos).
    zcr = librosa.feature.zero_crossing_rate(segmento, hop_length=config.HOP)

    # Spectral Centroid: "centro de masa" del espectro (en Hz).
    # Alto = sonido brillante; bajo = sonido grave u oscuro.
    centroide = librosa.feature.spectral_centroid(y=segmento, sr=sr, **kw)

    # Spectral Bandwidth: qué tan dispersas están las frecuencias alrededor del
    # centroide (ancho de banda).
    ancho = librosa.feature.spectral_bandwidth(y=segmento, sr=sr, **kw)

    # Spectral Rolloff: frecuencia por debajo de la cual está el 85 % de la
    # energía del espectro.
    rolloff = librosa.feature.spectral_rolloff(y=segmento, sr=sr, **kw)

    # RMS: energía/intensidad (volumen) de la señal.
    rms = librosa.feature.rms(y=segmento, frame_length=config.N_FFT,
                              hop_length=config.HOP)

    vector = np.concatenate([_estadisticos(m) for m in
                             (mfcc, chroma, zcr, centroide, ancho, rolloff, rms)])
    return vector.astype(np.float32)


def nombres_caracteristicas():
    """Nombre de cada posición del vector (para la tabla/CSV y la documentación)."""
    nombres = []
    grupos = [("mfcc", config.N_MFCC), ("chroma", 12), ("zcr", 1),
              ("centroide", 1), ("ancho_banda", 1), ("rolloff", 1), ("rms", 1)]
    for nombre, n in grupos:
        for est in ("media", "std"):
            if n == 1:
                nombres.append(f"{nombre}_{est}")
            else:
                nombres += [f"{nombre}{i + 1}_{est}" for i in range(n)]
    return nombres


# =====================================================================
# 3. MEL-ESPECTROGRAMA (entrada de la CNN)
# =====================================================================
def mel_db(segmento, n_mels=config.N_MELS_CNN):
    """
    Mel-espectrograma en decibeles.
      eje X = tiempo, eje Y = bandas Mel (frecuencia), valor = energía (dB).
    Es como una "imagen" del sonido que una CNN puede analizar.
    """
    mel = librosa.feature.melspectrogram(
        y=segmento, sr=config.SR, n_fft=config.N_FFT,
        hop_length=config.HOP, n_mels=n_mels)
    return librosa.power_to_db(mel, ref=1.0).astype(np.float32)


def mel_db_fijo(segmento):
    """Mel-espectrograma de tamaño fijo (N_MELS_CNN x FRAMES_MEL) para la CNN."""
    mel = mel_db(segmento)
    f = config.FRAMES_MEL
    if mel.shape[1] >= f:
        return mel[:, :f]
    # Segmento más corto de lo normal: se rellena con el valor mínimo (silencio)
    return np.pad(mel, ((0, 0), (0, f - mel.shape[1])),
                  mode="constant", constant_values=mel.min())


# =====================================================================
# 4. EJECUCIÓN DIRECTA: demostración del procesamiento de un audio
# =====================================================================
def _buscar_audio_ejemplo():
    for g in config.GENEROS:
        carpeta = config.DIR_DATASET / g
        if carpeta.exists():
            for f in sorted(carpeta.iterdir()):
                if f.suffix.lower() in config.EXTENSIONES:
                    return f
    return None


def main():
    parser = argparse.ArgumentParser(description="Muestra el procesamiento de un audio.")
    parser.add_argument("audio", nargs="?", help="Ruta del audio (opcional).")
    args = parser.parse_args()

    ruta = Path(args.audio) if args.audio else _buscar_audio_ejemplo()
    if ruta is None:
        print("No se indicó audio y el dataset está vacío. Uso:\n"
              "  python extraer_caracteristicas.py ruta\\audio.wav")
        return
    config.crear_carpetas()
    try:
        audio, sr = cargar_audio(ruta, duracion_max=config.DURACION_MAX_PREDICCION)
    except ErrorAudio as e:
        print("ERROR:", e)
        return

    print(f"\nArchivo               : {ruta.name}")
    print(f"Frecuencia de muestreo: {sr} Hz")
    print(f"Número de muestras    : {len(audio)}")
    print(f"Duración              : {len(audio) / sr:.2f} s")
    print(f"Amplitud mínima/máx.  : {audio.min():.3f} / {audio.max():.3f}")

    segmentos = dividir_segmentos(audio)
    vector = extraer_vector(segmentos[0])
    tabla = pd.DataFrame({"caracteristica": nombres_caracteristicas(), "valor": vector})
    print(f"\nSegmentos de {config.SEGMENTO_SEG:.0f} s: {len(segmentos)}")
    print(f"Tamaño del vector de características: {vector.shape[0]} números")
    print(tabla.to_string(index=False))

    base = ruta.stem
    tabla.to_csv(config.DIR_RESULTADOS / f"tabla_caracteristicas_{base}.csv", index=False)

    # Gráficas (fondo claro, pensadas para el informe)
    import visualizacion as vz
    for nombre, fig in (("forma_onda", vz.fig_forma_onda(audio, sr, oscuro=False)),
                        ("espectrograma", vz.fig_espectrograma(audio, sr, oscuro=False)),
                        ("mel_espectrograma", vz.fig_mel(audio, sr, oscuro=False))):
        destino = config.DIR_GRAFICAS / f"{nombre}_{base}.png"
        fig.savefig(destino, dpi=150)
        print("Gráfica guardada:", destino)
    print("Tabla guardada   :", config.DIR_RESULTADOS / f"tabla_caracteristicas_{base}.csv")


if __name__ == "__main__":
    main()
