"""
predecir.py
===========
Predice el género de un audio NUEVO usando el modelo ya entrenado.

El audio pasa por EXACTAMENTE el mismo procesamiento del entrenamiento:

    audio -> segmentos de 3 s -> extraer_vector() -> scaler.transform()
          -> red neuronal (Softmax) -> probabilidades por segmento
          -> promedio -> género con mayor probabilidad (+ confianza)

Uso por consola:
    python predecir.py ruta\\audio.wav
    python predecir.py ruta\\audio.wav --modelo cnn
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np

import config
import extraer_caracteristicas as ec


class ErrorModelo(Exception):
    """El modelo, el scaler o las etiquetas no existen o no son compatibles."""


class ClasificadorGeneros:
    """Carga el modelo entrenado y clasifica audios."""

    def __init__(self, tipo="mlp"):
        self.tipo = tipo
        faltan = [p for p in self._archivos_requeridos(tipo) if not p.exists()]
        if faltan:
            script = "entrenar_modelo.py" if tipo == "mlp" else "entrenar_cnn.py"
            raise ErrorModelo(
                "No se encontró el modelo entrenado:\n"
                + "\n".join(f" - {p.name}" for p in faltan)
                + f"\n\nEjecuta primero:  python {script}")
        try:
            from tensorflow import keras
            if tipo == "mlp":
                self.modelo = keras.models.load_model(config.RUTA_MODELO_MLP)
                self.scaler = joblib.load(config.RUTA_SCALER)
                self.etiquetas = joblib.load(config.RUTA_ETIQUETAS)
            else:
                self.modelo = keras.models.load_model(config.RUTA_MODELO_CNN)
                norm = joblib.load(config.RUTA_NORM_CNN)
                self.media, self.std = norm["media"], norm["std"]
                self.etiquetas = norm["etiquetas"]
        except Exception as e:
            raise ErrorModelo(f"No se pudo cargar el modelo:\n{e}") from e

    @staticmethod
    def _archivos_requeridos(tipo):
        if tipo == "mlp":
            return [config.RUTA_MODELO_MLP, config.RUTA_SCALER, config.RUTA_ETIQUETAS]
        return [config.RUTA_MODELO_CNN, config.RUTA_NORM_CNN]

    @staticmethod
    def modelos_disponibles():
        """Lista los tipos de modelo ya entrenados: ['mlp', 'cnn']."""
        return [t for t in ("mlp", "cnn")
                if all(p.exists() for p in ClasificadorGeneros._archivos_requeridos(t))]

    # ------------------------------------------------------------------
    def predecir_archivo(self, ruta):
        audio, sr = ec.cargar_audio(ruta, duracion_max=config.DURACION_MAX_PREDICCION)
        return self.predecir_senal(audio, sr)

    def predecir_senal(self, audio, sr):
        segmentos = ec.dividir_segmentos(audio)
        if self.tipo == "mlp":
            X = np.stack([ec.extraer_vector(s) for s in segmentos])
            if X.shape[1] != self.scaler.n_features_in_:
                raise ErrorModelo("El número de características no coincide con el "
                                  "usado en el entrenamiento. Reentrena el modelo.")
            X = self.scaler.transform(X)            # MISMO scaler del entrenamiento
        else:
            mel = np.stack([ec.mel_db_fijo(s) for s in segmentos])
            X = ((mel - self.media) / self.std)[..., np.newaxis]

        prob_segmentos = self.modelo.predict(X, verbose=0)    # una fila por segmento
        prob = prob_segmentos.mean(axis=0)                    # promedio -> audio completo
        i = int(np.argmax(prob))
        return dict(
            genero=self.etiquetas[i],
            confianza=float(prob[i] * 100),
            probabilidades={g: float(p) for g, p in zip(self.etiquetas, prob)},
            n_segmentos=len(segmentos),
            duracion=len(audio) / sr,
            audio=audio, sr=sr, modelo=self.tipo.upper())


def main():
    parser = argparse.ArgumentParser(description="Predice el género de un audio.")
    parser.add_argument("audio")
    parser.add_argument("--modelo", choices=["mlp", "cnn"], default="mlp")
    args = parser.parse_args()
    try:
        r = ClasificadorGeneros(args.modelo).predecir_archivo(args.audio)
    except (ec.ErrorAudio, ErrorModelo) as e:
        print("ERROR:", e)
        return
    print(f"\nModelo: {r['modelo']}  |  Segmentos analizados: {r['n_segmentos']}")
    print(f"\nGÉNERO DETECTADO: {r['genero'].upper()}")
    print(f"Confianza: {r['confianza']:.2f} %\n")
    for g, p in sorted(r["probabilidades"].items(), key=lambda kv: -kv[1]):
        print(f"  {g:<10} {p * 100:6.2f} %")


if __name__ == "__main__":
    main()
