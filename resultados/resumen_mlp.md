# Resultados reales del MLP

Resumen generado automáticamente por el script de entrenamiento con los resultados reales de esta ejecución.

## Parámetros

- **Características de entrada**: 74
- **Clases**: 10
- **Segmentos train/val/test**: 6981/1500/1500
- **Archivos en prueba**: 150
- **Batch size**: 32
- **Learning rate inicial**: 0.001
- **Optimizador**: Adam
- **Pérdida**: sparse_categorical_crossentropy
- **Épocas realmente entrenadas**: 28

## Evaluación en prueba - por segmento

Muestras evaluadas: 1500

| Métrica | Valor |
|---|---|
| Accuracy | 0.6920 |
| Precision (macro) | 0.6981 |
| Recall (macro) | 0.6920 |
| F1-score (macro) | 0.6908 |
| F1-score (ponderado) | 0.6908 |

```
              precision    recall  f1-score   support

       blues       0.67      0.69      0.68       150
   classical       0.89      0.93      0.91       150
     country       0.54      0.68      0.60       150
       disco       0.72      0.54      0.62       150
      hiphop       0.71      0.69      0.70       150
        jazz       0.85      0.76      0.80       150
       metal       0.84      0.79      0.81       150
         pop       0.64      0.71      0.67       150
      reggae       0.58      0.71      0.64       150
        rock       0.53      0.41      0.47       150

    accuracy                           0.69      1500
   macro avg       0.70      0.69      0.69      1500
weighted avg       0.70      0.69      0.69      1500

```

Confusiones más frecuentes (real -> predicho: cantidad):

- blues -> country: 28
- disco -> pop: 27
- rock -> country: 27
- hiphop -> reggae: 25
- rock -> pop: 22
- disco -> hiphop: 17

## Evaluación en prueba - por archivo

Muestras evaluadas: 150

| Métrica | Valor |
|---|---|
| Accuracy | 0.7533 |
| Precision (macro) | 0.7625 |
| Recall (macro) | 0.7533 |
| F1-score (macro) | 0.7489 |
| F1-score (ponderado) | 0.7489 |

```
              precision    recall  f1-score   support

       blues       0.79      0.73      0.76        15
   classical       0.88      1.00      0.94        15
     country       0.71      0.80      0.75        15
       disco       0.80      0.53      0.64        15
      hiphop       0.80      0.80      0.80        15
        jazz       0.92      0.73      0.81        15
       metal       0.82      0.93      0.88        15
         pop       0.63      0.80      0.71        15
      reggae       0.58      0.73      0.65        15
        rock       0.70      0.47      0.56        15

    accuracy                           0.75       150
   macro avg       0.76      0.75      0.75       150
weighted avg       0.76      0.75      0.75       150

```

Confusiones más frecuentes (real -> predicho: cantidad):

- rock -> pop: 4
- disco -> pop: 3
- blues -> country: 2
- blues -> reggae: 2
- disco -> hiphop: 2
- hiphop -> reggae: 2
