from pathlib import Path

# ---------------------------------------------------------------- Rutas
RAIZ = Path(__file__).resolve().parent
DIR_DATASET = RAIZ / "dataset"
DIR_MODELO = RAIZ / "modelo"
DIR_RESULTADOS = RAIZ / "resultados"
DIR_GRAFICAS = DIR_RESULTADOS / "graficas"
DIR_MATRICES = DIR_RESULTADOS / "matrices"
DIR_PREDICCIONES = DIR_RESULTADOS / "predicciones"
DIR_CACHE = RAIZ / "cache"

RUTA_MODELO_MLP = DIR_MODELO / "modelo_generos.keras"
RUTA_SCALER = DIR_MODELO / "scaler.pkl"
RUTA_ETIQUETAS = DIR_MODELO / "etiquetas.pkl"
RUTA_MODELO_CNN = DIR_MODELO / "modelo_cnn.keras"
RUTA_NORM_CNN = DIR_MODELO / "cnn_normalizacion.pkl"

RUTA_CSV_CARACT = DIR_RESULTADOS / "caracteristicas_dataset.csv"
RUTA_NPZ_MEL = DIR_CACHE / "mel_segmentos.npz"
RUTA_RESUMEN_DATASET = DIR_RESULTADOS / "resumen_dataset.json"
RUTA_PROBLEMAS = DIR_RESULTADOS / "archivos_problematicos.csv"
RUTA_SPLIT = DIR_RESULTADOS / "division_archivos.csv"

# ---------------------------------------------------------------- Géneros GTZAN
GENEROS = ["blues", "classical", "country", "disco", "hiphop",
           "jazz", "metal", "pop", "reggae", "rock"]

# Formatos que se aceptan (el GTZAN original viene en .au; Kaggle lo trae en .wav)
EXTENSIONES = (".wav", ".mp3", ".flac", ".ogg", ".au")

# ---------------------------------------------------------------- Audio
SR = 22050                    # frecuencia de muestreo (muestras por segundo)
SEGMENTO_SEG = 3.0            # cada audio se divide en ventanas de 3 s
DURACION_MAX_ENTRENAMIENTO = 30.0   # los clips de GTZAN duran ~30 s
DURACION_MAX_PREDICCION = 60.0      # en la app solo se analizan los primeros 60 s
DURACION_MIN_SEG = 1.0        # menos de 1 s se considera demasiado corto
UMBRAL_SILENCIO = 1e-4        # amplitud máxima por debajo de esto = silencio

# ---------------------------------------------------------------- Características
N_MFCC = 20
N_FFT = 2048                  # tamaño de ventana de la FFT
HOP = 512                     # salto entre ventanas
N_MELS_CNN = 64               # filas del Mel-espectrograma que recibe la CNN
FRAMES_MEL = 1 + int(SEGMENTO_SEG * SR) // HOP    # columnas (130 con estos valores)
N_MELS_VISUAL = 128           # resolución del Mel que se muestra en pantalla

# ---------------------------------------------------------------- Entrenamiento
SEMILLA = 42
PROP_VAL = 0.15
PROP_TEST = 0.15

# ---------------------------------------------------------------- Micrófono
MIC_DURACION_DEFECTO = 10     # segundos
MIC_DURACION_MIN = 3
MIC_DURACION_MAX = 30


def crear_carpetas():
    """Crea las carpetas de salida si todavía no existen."""
    for d in (DIR_DATASET, DIR_MODELO, DIR_GRAFICAS, DIR_MATRICES,
              DIR_PREDICCIONES, DIR_CACHE):
        d.mkdir(parents=True, exist_ok=True)
