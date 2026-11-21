# Preparación para entrega por GitHub

## Contenido que se versiona

El repositorio incluye el código fuente, el modelo MLP, el scaler, el mapa de
etiquetas, la tabla de características, métricas, matrices, gráficas,
documentación y evidencia de predicción. Esto permite revisar el proyecto y
ejecutar una inferencia sin reentrenar.

## Contenido excluido

`dataset/` y `cache/` se excluyen porque contienen archivos voluminosos o
regenerables. `.venv/`, copias de seguridad, registros temporales y grabaciones
personales también se excluyen. Consulte `.gitignore`.

## Publicación

1. Cree un repositorio vacío en GitHub, sin README, licencia ni `.gitignore`
   iniciales.
2. En esta carpeta ejecute:

```powershell
git add .
git commit -m "Entrega final clasificador de géneros musicales"
git branch -M main
git remote add origin URL_DEL_REPOSITORIO
git push -u origin main
```

3. Revise en GitHub que no se hayan incluido `dataset/`, `cache/`, `.venv/` ni
   `respaldo_preparacion_github_2026-11-20/`.

## Reproducción para quien revise

1. Clone el repositorio.
2. Cree y active un entorno de Python 3.12.
3. Ejecute `pip install -r requirements.txt`.
4. Para inferir con los artefactos incluidos, ejecute `python app.py` o
   `python predecir.py RUTA\AUDIO.wav`.
5. Para reconstruir el dataset o reentrenar, obtenga GTZAN desde una fuente
   autorizada, colóquelo en `dataset/` y siga el README. Ese paso es opcional y
   puede reemplazar los artefactos incluidos.
