"""
preparar_dataset.py
===================
Recorre dataset/<genero>/*.wav, extrae las características de cada audio y
construye las tablas de entrenamiento:

    X = características (vector numérico por segmento de 3 s)
    y = género (se conserva en la columna 'genero' junto al nombre del archivo)

La correspondencia audio <-> género se mantiene porque la etiqueta sale del
NOMBRE DE LA CARPETA y se guarda en la misma fila que sus características.

Salidas (todas generadas con los datos reales):
    resultados/caracteristicas_dataset.csv  tabla de características
    cache/mel_segmentos.npz                 Mel-espectrogramas (para la CNN)
    resultados/resumen_dataset.json         conteos reales
    resultados/archivos_problematicos.csv   archivos que no se pudieron procesar

Uso:
    python preparar_dataset.py            (calcula también los Mel para la CNN)
    python preparar_dataset.py --sin-mel  (solo MLP, más rápido)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config
import extraer_caracteristicas as ec


def listar_archivos(dir_dataset):
    """Devuelve [(ruta, genero)] y la lista de carpetas de género que faltan."""
    archivos, faltantes = [], []
    for genero in config.GENEROS:
        carpeta = dir_dataset / genero
        if not carpeta.is_dir():
            faltantes.append(genero)
            continue
        for f in sorted(carpeta.iterdir()):
            if f.is_file() and f.suffix.lower() in config.EXTENSIONES:
                archivos.append((f, genero))
    return archivos, faltantes


def procesar_archivo(ruta, genero, dir_dataset, con_mel):
    """
    Procesa UN archivo. Devuelve un diccionario con 'ok' (True/False).
    Los errores no detienen el proceso: el archivo se registra como problemático
    (en GTZAN, por ejemplo, jazz.00054 suele venir dañado).
    """
    ident = ruta.relative_to(dir_dataset).as_posix()
    try:
        audio, _ = ec.cargar_audio(ruta, duracion_max=config.DURACION_MAX_ENTRENAMIENTO)
        segmentos = ec.dividir_segmentos(audio)
        vectores = np.stack([ec.extraer_vector(s) for s in segmentos])
        if not np.all(np.isfinite(vectores)):
            raise ec.ErrorAudio("Las características contienen NaN/infinito.")
        mels = np.stack([ec.mel_db_fijo(s) for s in segmentos]) if con_mel else None
        return dict(ok=True, archivo=ident, genero=genero, vectores=vectores, mels=mels)
    except Exception as e:
        return dict(ok=False, archivo=ident, genero=genero, error=str(e).replace("\n", " "))


def main():
    parser = argparse.ArgumentParser(description="Prepara el dataset de características.")
    parser.add_argument("--dataset", default=str(config.DIR_DATASET))
    parser.add_argument("--jobs", type=int, default=-1,
                        help="Procesos en paralelo (-1 = todos los núcleos).")
    parser.add_argument("--sin-mel", action="store_true",
                        help="No generar Mel-espectrogramas (solo MLP).")
    args = parser.parse_args()

    config.crear_carpetas()
    dir_dataset = Path(args.dataset)
    con_mel = not args.sin_mel

    archivos, faltantes = listar_archivos(dir_dataset)
    if faltantes:
        print("AVISO: faltan las carpetas de género:", ", ".join(faltantes))
    if not archivos:
        print(f"\nNo se encontró ningún audio en {dir_dataset}.\n"
              "Coloca GTZAN así: dataset\\blues\\blues.00000.wav, dataset\\rock\\...")
        return

    print(f"Archivos de audio encontrados: {len(archivos)}")
    print("Extrayendo características (puede tardar varios minutos)...")
    t0 = time.time()
    resultados = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(procesar_archivo)(r, g, dir_dataset, con_mel) for r, g in archivos)

    buenos = [r for r in resultados if r["ok"]]
    malos = [r for r in resultados if not r["ok"]]
    if not buenos:
        print("Ningún archivo pudo procesarse. Revisa los formatos.")
        return

    # ---- Construcción de la tabla: una fila por segmento de 3 s
    filas_meta, bloques, mels = [], [], []
    for r in buenos:
        n = len(r["vectores"])
        filas_meta += [(r["archivo"], r["genero"], i) for i in range(n)]
        bloques.append(r["vectores"])
        if con_mel:
            mels.append(r["mels"])
    X = np.vstack(bloques)
    meta = pd.DataFrame(filas_meta, columns=["archivo", "genero", "segmento"])
    tabla = pd.concat([meta, pd.DataFrame(X, columns=ec.nombres_caracteristicas())], axis=1)
    tabla.to_csv(config.RUTA_CSV_CARACT, index=False)

    if con_mel:
        np.savez_compressed(config.RUTA_NPZ_MEL,
                            mel=np.concatenate(mels).astype(np.float16),
                            archivo=meta["archivo"].values.astype(str),
                            genero=meta["genero"].values.astype(str))

    pd.DataFrame([(r["archivo"], r["genero"], r["error"]) for r in malos],
                 columns=["archivo", "genero", "error"]).to_csv(
        config.RUTA_PROBLEMAS, index=False)

    resumen = dict(
        archivos_encontrados=len(archivos),
        archivos_procesados=len(buenos),
        archivos_problematicos=len(malos),
        segmentos_totales=int(len(tabla)),
        numero_caracteristicas=int(X.shape[1]),
        archivos_por_genero=meta.drop_duplicates("archivo")["genero"].value_counts().to_dict(),
        segundos_proceso=round(time.time() - t0, 1),
    )
    config.RUTA_RESUMEN_DATASET.write_text(json.dumps(resumen, indent=2, ensure_ascii=False),
                                           encoding="utf-8")

    print("\n===== RESUMEN REAL DEL DATASET =====")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    if malos:
        print("\nArchivos problemáticos (omitidos):")
        for r in malos:
            print(" -", r["archivo"], "->", r["error"][:90])
    print("\nTabla de características:", config.RUTA_CSV_CARACT)
    if con_mel:
        print("Mel-espectrogramas      :", config.RUTA_NPZ_MEL)
    print("Siguiente paso: python entrenar_modelo.py")


if __name__ == "__main__":
    main()
