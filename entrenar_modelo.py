"""
entrenar_modelo.py
==================
Entrena el modelo principal: una red neuronal MLP (Multi-Layer Perceptron)
que recibe el vector de características numéricas de un audio.

Flujo:
    CSV de características -> X, y -> división train/val/test (por archivo)
    -> normalización (scaler SOLO con train) -> MLP -> entrenamiento
    -> evaluación -> guardar modelo, scaler y etiquetas -> gráficas/métricas

Uso:   python entrenar_modelo.py
       python entrenar_modelo.py --epocas 200 --batch 64 --lr 0.0005
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # menos mensajes de TensorFlow

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
import utilidades_modelo as um


def construir_mlp(n_entradas, n_clases):
    """
    Arquitectura MLP:

        Entrada (74 características)
           -> Dense(128) + ReLU  -> Dropout(0.3)
           -> Dense(64)  + ReLU  -> Dropout(0.3)
           -> Dense(n_clases) + Softmax   (10 géneros)

    * Neurona : calcula una suma ponderada de sus entradas (+ sesgo) y aplica
                una función de activación.
    * Dense   : capa donde CADA neurona se conecta con TODAS las de la capa previa.
    * ReLU    : f(x) = max(0, x). Introduce no linealidad (sin ella, apilar capas
                equivaldría a una sola) y es simple y eficiente de entrenar.
    * Dropout : durante el entrenamiento "apaga" al azar un 30 % de neuronas para
                reducir el sobreajuste (overfitting).
    * Softmax : convierte las 10 salidas en probabilidades que suman 1.
                La mayor es el género predicho; su valor es la "confianza".
    * Se eligieron 128 y 64 neuronas: suficientes para ~74 entradas, con pocos
      parámetros (se entrena rápido en CPU y es fácil de explicar).
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    return keras.Sequential([
        layers.Input(shape=(n_entradas,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_clases, activation="softmax"),
    ], name="mlp_generos")


def main():
    parser = argparse.ArgumentParser(description="Entrena el MLP de géneros musicales.")
    parser.add_argument("--epocas", type=int, default=150, help="Máximo de épocas.")
    parser.add_argument("--batch", type=int, default=32, help="Tamaño de lote (batch size).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    args = parser.parse_args()

    config.crear_carpetas()
    if not config.RUTA_CSV_CARACT.exists():
        print("No existe", config.RUTA_CSV_CARACT,
              "\nPrimero ejecuta:  python preparar_dataset.py")
        return

    from tensorflow import keras
    keras.utils.set_random_seed(config.SEMILLA)       # resultados reproducibles

    # ------------------------------------------------ 1. Datos: X (características) e y (género)
    df = pd.read_csv(config.RUTA_CSV_CARACT)
    columnas_x = [c for c in df.columns if c not in ("archivo", "genero", "segmento")]
    antes = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=columnas_x)   # limpieza
    if len(df) < antes:
        print(f"Limpieza: se eliminaron {antes - len(df)} filas con valores inválidos.")

    etiquetas = sorted(df["genero"].unique())          # orden alfabético fijo
    if len(etiquetas) < 2:
        print("Se necesitan al menos 2 géneros para entrenar.")
        return
    indice = {g: i for i, g in enumerate(etiquetas)}
    X = df[columnas_x].values.astype(np.float32)
    y = df["genero"].map(indice).values
    print(f"X: {X.shape}  (segmentos x características)   y: {y.shape}   clases: {len(etiquetas)}")

    # ------------------------------------------------ 2. División por archivo
    gen_por_archivo = df.drop_duplicates("archivo").set_index("archivo")["genero"].to_dict()
    mapa = um.dividir_archivos(df["archivo"], df["genero"])
    um.guardar_division(mapa, gen_por_archivo)
    particion = df["archivo"].map(mapa).values
    m_tr, m_va, m_te = (particion == "train"), (particion == "val"), (particion == "test")
    print(f"Segmentos -> entrenamiento: {m_tr.sum()} | validación: {m_va.sum()} | prueba: {m_te.sum()}")

    # ------------------------------------------------ 3. Normalización SIN fuga de información
    # El scaler aprende media y desviación SOLO del entrenamiento. Luego se
    # aplica igual a validación, prueba y a los audios nuevos de la aplicación.
    scaler = StandardScaler().fit(X[m_tr])
    X_tr, X_va, X_te = (scaler.transform(X[m]) for m in (m_tr, m_va, m_te))
    y_tr, y_va, y_te = y[m_tr], y[m_va], y[m_te]

    # ------------------------------------------------ 4. Modelo
    modelo = construir_mlp(X.shape[1], len(etiquetas))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),   # cómo se ajustan los pesos
        loss="sparse_categorical_crossentropy",                   # error para clases enteras
        metrics=["accuracy"])
    modelo.summary()

    # ------------------------------------------------ 5. Entrenamiento
    # Época = una pasada completa por todos los datos de entrenamiento.
    # Batch size = cuántas muestras se procesan antes de actualizar los pesos.
    # EarlyStopping: detiene el entrenamiento si la pérdida de validación deja de
    # mejorar y conserva los mejores pesos (evita sobreajuste).
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6),
    ]
    hist = modelo.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=args.epocas,
                      batch_size=args.batch, callbacks=callbacks, verbose=2)

    # ------------------------------------------------ 6. Evaluación en PRUEBA (datos no usados)
    prob_te = modelo.predict(X_te, verbose=0)
    m_seg = um.evaluar(y_te, prob_te.argmax(axis=1), etiquetas, "mlp_segmentos",
                       "MLP (por segmento de 3 s)")
    P_arch, y_arch = um.agregar_por_archivo(df["archivo"].values[m_te], prob_te, y_te)
    m_arch = um.evaluar(y_arch, P_arch.argmax(axis=1), etiquetas, "mlp_archivos",
                        "MLP (por archivo completo)")

    # ------------------------------------------------ 7. Guardado
    modelo.save(config.RUTA_MODELO_MLP)
    joblib.dump(scaler, config.RUTA_SCALER)
    joblib.dump(etiquetas, config.RUTA_ETIQUETAS)
    um.graficar_historial(hist.history, "mlp", "MLP")
    n_ep = len(hist.history["loss"])
    um.escribir_resumen(
        config.DIR_RESULTADOS / "resumen_mlp.md", "Resultados reales del MLP",
        {"Características de entrada": X.shape[1], "Clases": len(etiquetas),
         "Segmentos train/val/test": f"{m_tr.sum()}/{m_va.sum()}/{m_te.sum()}",
         "Archivos en prueba": len(y_arch), "Batch size": args.batch,
         "Learning rate inicial": args.lr, "Optimizador": "Adam",
         "Pérdida": "sparse_categorical_crossentropy"},
        {"por segmento": m_seg, "por archivo": m_arch}, n_ep)
    um.guardar_json(config.DIR_RESULTADOS / "metricas_mlp.json",
                    {"segmentos": {k: v for k, v in m_seg.items() if k != "reporte"},
                     "archivos": {k: v for k, v in m_arch.items() if k != "reporte"}})

    print("\n===== RESULTADOS REALES (conjunto de prueba) =====")
    print(f"Accuracy por segmento: {m_seg['accuracy']:.4f}")
    print(f"Accuracy por archivo : {m_arch['accuracy']:.4f}")
    print(m_arch["reporte"])
    print("Modelo   :", config.RUTA_MODELO_MLP)
    print("Scaler   :", config.RUTA_SCALER)
    print("Etiquetas:", config.RUTA_ETIQUETAS)
    print("Resumen  :", config.DIR_RESULTADOS / "resumen_mlp.md")


if __name__ == "__main__":
    main()
