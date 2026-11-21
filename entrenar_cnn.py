"""
entrenar_cnn.py  (EXTENSIÓN del proyecto)
=========================================
Entrena una CNN (Red Neuronal Convolucional) que recibe el Mel-espectrograma
de cada segmento de 3 s, tratándolo como una imagen:

    Audio -> Mel-espectrograma (64 x 130) -> CNN -> género

Requisito: haber ejecutado antes  python preparar_dataset.py  (sin --sin-mel).
Usa la MISMA división por archivo que el MLP para poder compararlos.

Uso:   python entrenar_cnn.py
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np

import config
import utilidades_modelo as um


def construir_cnn(forma_entrada, n_clases):
    """
    Entrada: Mel-espectrograma (64 bandas Mel x 130 instantes x 1 canal).

        [Conv2D + ReLU + BatchNorm + MaxPool] x 3   (16 -> 32 -> 64 filtros)
        Flatten -> Dropout -> Dense(64)+ReLU -> Dropout -> Dense(n)+Softmax

    * Conv2D: desliza pequeños filtros (3x3) sobre la "imagen" para detectar
      patrones locales (p. ej. líneas de armónicos, golpes de batería).
    * MaxPool: reduce el tamaño conservando lo más importante.
    * Una CNN aprovecha la estructura bidimensional (tiempo-frecuencia) que el
      MLP pierde al recibir solo promedios y desviaciones.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    return keras.Sequential([
        layers.Input(shape=forma_entrada),
        layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_clases, activation="softmax"),
    ], name="cnn_generos")


def main():
    parser = argparse.ArgumentParser(description="Entrena la CNN con Mel-espectrogramas.")
    parser.add_argument("--epocas", type=int, default=60)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    config.crear_carpetas()
    if not config.RUTA_NPZ_MEL.exists():
        print("No existe", config.RUTA_NPZ_MEL,
              "\nEjecuta primero:  python preparar_dataset.py   (sin --sin-mel)")
        return

    from tensorflow import keras
    keras.utils.set_random_seed(config.SEMILLA)

    datos = np.load(config.RUTA_NPZ_MEL)
    mel, archivos, generos = datos["mel"], datos["archivo"], datos["genero"]
    etiquetas = sorted(set(generos.tolist()))
    indice = {g: i for i, g in enumerate(etiquetas)}
    y = np.array([indice[g] for g in generos])
    print(f"Mel-espectrogramas: {mel.shape}  clases: {len(etiquetas)}")

    mapa = um.dividir_archivos(archivos, generos)        # misma división que el MLP
    particion = np.array([mapa[a] for a in archivos])
    m_tr, m_va, m_te = (particion == "train"), (particion == "val"), (particion == "test")

    # Normalización con estadísticos SOLO del entrenamiento (sin fuga)
    media = float(mel[m_tr].astype(np.float32).mean())
    std = float(mel[m_tr].astype(np.float32).std()) + 1e-8

    def preparar(m):
        return ((mel[m].astype(np.float32) - media) / std)[..., np.newaxis]

    X_tr, X_va, X_te = preparar(m_tr), preparar(m_va), preparar(m_te)
    y_tr, y_va, y_te = y[m_tr], y[m_va], y[m_te]
    print(f"Segmentos -> entrenamiento: {len(y_tr)} | validación: {len(y_va)} | prueba: {len(y_te)}")

    modelo = construir_cnn(X_tr.shape[1:], len(etiquetas))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    modelo.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
    ]
    hist = modelo.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=args.epocas,
                      batch_size=args.batch, callbacks=callbacks, verbose=2)

    prob_te = modelo.predict(X_te, batch_size=64, verbose=0)
    m_seg = um.evaluar(y_te, prob_te.argmax(axis=1), etiquetas, "cnn_segmentos",
                       "CNN (por segmento de 3 s)")
    P_arch, y_arch = um.agregar_por_archivo(archivos[m_te], prob_te, y_te)
    m_arch = um.evaluar(y_arch, P_arch.argmax(axis=1), etiquetas, "cnn_archivos",
                        "CNN (por archivo completo)")

    modelo.save(config.RUTA_MODELO_CNN)
    joblib.dump({"media": media, "std": std, "etiquetas": etiquetas}, config.RUTA_NORM_CNN)
    um.graficar_historial(hist.history, "cnn", "CNN")
    um.escribir_resumen(
        config.DIR_RESULTADOS / "resumen_cnn.md", "Resultados reales de la CNN",
        {"Entrada": f"Mel {X_tr.shape[1]}x{X_tr.shape[2]}", "Clases": len(etiquetas),
         "Segmentos train/val/test": f"{len(y_tr)}/{len(y_va)}/{len(y_te)}",
         "Archivos en prueba": len(y_arch), "Batch size": args.batch,
         "Learning rate inicial": args.lr, "Optimizador": "Adam"},
        {"por segmento": m_seg, "por archivo": m_arch}, len(hist.history["loss"]))
    um.guardar_json(config.DIR_RESULTADOS / "metricas_cnn.json",
                    {"segmentos": {k: v for k, v in m_seg.items() if k != "reporte"},
                     "archivos": {k: v for k, v in m_arch.items() if k != "reporte"}})

    print("\n===== RESULTADOS REALES CNN (conjunto de prueba) =====")
    print(f"Accuracy por segmento: {m_seg['accuracy']:.4f}")
    print(f"Accuracy por archivo : {m_arch['accuracy']:.4f}")
    print(m_arch["reporte"])
    print("Modelo CNN:", config.RUTA_MODELO_CNN)


if __name__ == "__main__":
    main()
