# Evidencia de clasificación con audio nuevo

## Entrada utilizada

- Archivo: `resultados/predicciones/grabacion_20261014_172759.wav`
- Tipo: grabación externa al dataset local.
- Verificación de independencia: su SHA-256 es
  `fe1f0ce1165d82d5969b53e999d9fffa3bd87ab2eff9654d9dd4a27b5fcc60c0` y no
  coincide con ningún WAV de `dataset/`.
- Comando ejecutado sin modificar código ni artefactos del modelo:

```powershell
python predecir.py resultados\predicciones\grabacion_20261014_172759.wav --modelo mlp
```

## Resultado obtenido

- Modelo: MLP existente.
- Segmentos analizados: 3.
- Género predicho: `classical`.
- Confianza: 52,97 %.

La salida completa de consola se conserva en
`resultados/predicciones/evidencia_prediccion_audio_nuevo.txt`. La confianza
es la probabilidad Softmax de la clase elegida; no debe interpretarse como una
garantía de que el género real sea ese.
