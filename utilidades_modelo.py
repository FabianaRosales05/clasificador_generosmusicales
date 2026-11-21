"""
utilidades_modelo.py
====================
Funciones compartidas por entrenar_modelo.py (MLP) y entrenar_cnn.py (CNN):
división de datos, métricas, gráficas y resumen en Markdown.
"""
import json

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

import config


# ---------------------------------------------------------------- División
def dividir_archivos(archivos, generos):
    """
    Divide POR ARCHIVO (no por segmento) en entrenamiento / validación / prueba.

    Por qué por archivo: los segmentos de una misma canción son muy parecidos.
    Si unos cayeran en entrenamiento y otros en prueba, el modelo "vería" la
    canción durante el entrenamiento y el resultado sería artificialmente alto
    (fuga de información). Aquí cada canción pertenece a UNA sola partición.
    """
    df = (pd.DataFrame({"archivo": archivos, "genero": generos})
          .drop_duplicates("archivo").sort_values("archivo").reset_index(drop=True))
    resto, prueba = train_test_split(df, test_size=config.PROP_TEST,
                                     stratify=df["genero"], random_state=config.SEMILLA)
    prop_val = config.PROP_VAL / (1 - config.PROP_TEST)
    entren, valid = train_test_split(resto, test_size=prop_val,
                                     stratify=resto["genero"], random_state=config.SEMILLA)
    mapa = {}
    for nombre, parte in (("train", entren), ("val", valid), ("test", prueba)):
        for a in parte["archivo"]:
            mapa[a] = nombre
    return mapa


def guardar_division(mapa, generos_por_archivo):
    df = pd.DataFrame({"archivo": list(mapa.keys()),
                       "particion": list(mapa.values())})
    df["genero"] = df["archivo"].map(generos_por_archivo)
    df.sort_values(["particion", "archivo"]).to_csv(config.RUTA_SPLIT, index=False)


# ---------------------------------------------------------------- Agregación
def agregar_por_archivo(archivos, probabilidades, y_real):
    """Promedia las probabilidades de todos los segmentos de cada archivo."""
    df = pd.DataFrame(probabilidades)
    df["archivo"] = np.asarray(archivos)
    df["y"] = np.asarray(y_real)
    g = df.groupby("archivo")
    cols = list(range(probabilidades.shape[1]))
    P = g[cols].mean()
    y = g["y"].first()
    return P.values, y.values.astype(int)


# ---------------------------------------------------------------- Gráficas
def graficar_historial(historial, prefijo, titulo):
    """Guarda pérdida y accuracy (entrenamiento vs validación) por época."""
    epocas = range(1, len(historial["loss"]) + 1)
    for clave, nombre, archivo in (("loss", "Pérdida", "perdida"),
                                   ("accuracy", "Accuracy", "accuracy")):
        fig = Figure(figsize=(7, 4.2), dpi=120, layout="constrained")
        ax = fig.add_subplot(111)
        ax.plot(epocas, historial[clave], label="Entrenamiento", linewidth=2)
        ax.plot(epocas, historial["val_" + clave], label="Validación", linewidth=2)
        ax.set_title(f"{titulo}: {nombre} por época")
        ax.set_xlabel("Época")
        ax.set_ylabel(nombre)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.savefig(config.DIR_GRAFICAS / f"{archivo}_{prefijo}.png")


def graficar_matriz(cm, etiquetas, ruta, titulo):
    fig = Figure(figsize=(8, 7), dpi=120, layout="constrained")
    ax = fig.add_subplot(111)
    img = ax.imshow(cm, cmap="Blues")
    fig.colorbar(img, ax=ax)
    ax.set_xticks(range(len(etiquetas)), labels=etiquetas, rotation=45, ha="right")
    ax.set_yticks(range(len(etiquetas)), labels=etiquetas)
    umbral = cm.max() / 2 if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > umbral else "black", fontsize=9)
    ax.set_title(titulo)
    ax.set_xlabel("Género predicho")
    ax.set_ylabel("Género real")
    fig.savefig(ruta)


# ---------------------------------------------------------------- Métricas
def evaluar(y_real, y_pred, etiquetas, prefijo, titulo):
    """Calcula accuracy, precision, recall, F1, reporte y matriz de confusión."""
    labels = list(range(len(etiquetas)))
    acc = accuracy_score(y_real, y_pred)
    pm, rm, fm, _ = precision_recall_fscore_support(y_real, y_pred, average="macro",
                                                    zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_real, y_pred, average="weighted",
                                                    zero_division=0)
    reporte = classification_report(y_real, y_pred, labels=labels,
                                    target_names=etiquetas, zero_division=0)
    cm = confusion_matrix(y_real, y_pred, labels=labels)

    (config.DIR_RESULTADOS / f"reporte_{prefijo}.txt").write_text(
        f"{titulo}\n\n{reporte}", encoding="utf-8")
    pd.DataFrame(cm, index=etiquetas, columns=etiquetas).to_csv(
        config.DIR_MATRICES / f"matriz_confusion_{prefijo}.csv")
    graficar_matriz(cm, etiquetas, config.DIR_MATRICES / f"matriz_confusion_{prefijo}.png",
                    f"Matriz de confusión - {titulo}")

    confusiones = []
    for i in labels:
        for j in labels:
            if i != j and cm[i, j] > 0:
                confusiones.append((etiquetas[i], etiquetas[j], int(cm[i, j])))
    confusiones.sort(key=lambda c: -c[2])

    return dict(accuracy=float(acc),
                precision_macro=float(pm), recall_macro=float(rm), f1_macro=float(fm),
                precision_ponderada=float(pw), recall_ponderado=float(rw),
                f1_ponderado=float(fw), n_muestras=int(len(y_real)),
                reporte=reporte, confusiones=confusiones[:6])


def escribir_resumen(ruta, titulo, parametros, metricas_por_nivel, epocas_entrenadas):
    """Genera un resumen Markdown con los números REALES de esta ejecución."""
    L = [f"# {titulo}", "", "Resumen generado automáticamente por el script de "
         "entrenamiento con los resultados reales de esta ejecución.", "",
         "## Parámetros", ""]
    L += [f"- **{k}**: {v}" for k, v in parametros.items()]
    L += [f"- **Épocas realmente entrenadas**: {epocas_entrenadas}", ""]
    for nivel, m in metricas_por_nivel.items():
        L += [f"## Evaluación en prueba - {nivel}", "",
              f"Muestras evaluadas: {m['n_muestras']}", "",
              "| Métrica | Valor |", "|---|---|",
              f"| Accuracy | {m['accuracy']:.4f} |",
              f"| Precision (macro) | {m['precision_macro']:.4f} |",
              f"| Recall (macro) | {m['recall_macro']:.4f} |",
              f"| F1-score (macro) | {m['f1_macro']:.4f} |",
              f"| F1-score (ponderado) | {m['f1_ponderado']:.4f} |", "",
              "```", m["reporte"], "```", "",
              "Confusiones más frecuentes (real -> predicho: cantidad):", ""]
        if m["confusiones"]:
            L += [f"- {a} -> {b}: {n}" for a, b, n in m["confusiones"]]
        else:
            L += ["- (no hubo confusiones)"]
        L.append("")
    ruta.write_text("\n".join(L), encoding="utf-8")


def guardar_json(ruta, datos):
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
