"""
visualizacion.py
================
Funciones que construyen las gráficas del audio y de la predicción.
Devuelven objetos Figure de Matplotlib (sirven para mostrarlos en la
interfaz Tkinter o para guardarlos como PNG).
"""
import numpy as np
import librosa
import librosa.display
from matplotlib.figure import Figure

import config


def _tema(oscuro):
    if oscuro:
        return dict(fondo="#1e293b", texto="#e2e8f0", rejilla="#334155",
                    acento="#818cf8", resalte="#22c55e", cmap="magma")
    return dict(fondo="white", texto="#111827", rejilla="#d1d5db",
                acento="#4f46e5", resalte="#16a34a", cmap="magma")


def _nueva_figura(titulo, oscuro, size=(5.2, 3.2)):
    t = _tema(oscuro)
    fig = Figure(figsize=size, dpi=100, facecolor=t["fondo"], layout="constrained")
    ax = fig.add_subplot(111)
    ax.set_facecolor(t["fondo"])
    ax.set_title(titulo, color=t["texto"], fontsize=11, fontweight="bold")
    ax.tick_params(colors=t["texto"], labelsize=8)
    for s in ax.spines.values():
        s.set_color(t["rejilla"])
    return fig, ax, t


def _etiquetas(ax, t, x, y):
    ax.set_xlabel(x, color=t["texto"], fontsize=9)
    ax.set_ylabel(y, color=t["texto"], fontsize=9)


def _barra_color(fig, mappable, ax, t, etiqueta="dB"):
    cb = fig.colorbar(mappable, ax=ax, format="%+2.0f dB")
    cb.ax.tick_params(colors=t["texto"], labelsize=7)
    cb.outline.set_edgecolor(t["rejilla"])
    cb.set_label(etiqueta, color=t["texto"], fontsize=8)


def fig_forma_onda(audio, sr, oscuro=True):
    """Forma de onda: AMPLITUD de la señal a lo largo del TIEMPO."""
    fig, ax, t = _nueva_figura("Forma de onda", oscuro)
    tiempo = np.arange(len(audio)) / sr
    paso = max(1, len(audio) // 20000)          # submuestreo solo para dibujar rápido
    ax.plot(tiempo[::paso], audio[::paso], color=t["acento"], linewidth=0.5)
    ax.set_xlim(0, tiempo[-1] if len(tiempo) else 1)
    _etiquetas(ax, t, "Tiempo (s)", "Amplitud")
    return fig


def fig_espectrograma(audio, sr, oscuro=True):
    """
    Espectrograma (STFT): cómo se distribuye la ENERGÍA en distintas FRECUENCIAS
    a lo largo del TIEMPO. Eje X = tiempo, eje Y = frecuencia, color = energía.
    """
    fig, ax, t = _nueva_figura("Espectrograma (STFT)", oscuro)
    stft = librosa.stft(audio, n_fft=config.N_FFT, hop_length=config.HOP)
    db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    img = librosa.display.specshow(db, sr=sr, hop_length=config.HOP,
                                   x_axis="time", y_axis="log", cmap=t["cmap"], ax=ax)
    _etiquetas(ax, t, "Tiempo (s)", "Frecuencia (Hz, escala log)")
    _barra_color(fig, img, ax, t)
    return fig


def fig_mel(audio, sr, oscuro=True):
    """
    Mel-espectrograma: como el espectrograma, pero el eje de frecuencia usa la
    escala Mel (más cercana a cómo oye el oído humano).
    """
    fig, ax, t = _nueva_figura("Mel-espectrograma", oscuro)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=config.N_FFT,
                                         hop_length=config.HOP,
                                         n_mels=config.N_MELS_VISUAL)
    db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(db, sr=sr, hop_length=config.HOP,
                                   x_axis="time", y_axis="mel", cmap=t["cmap"], ax=ax)
    _etiquetas(ax, t, "Tiempo (s)", "Frecuencia (Hz, escala Mel)")
    _barra_color(fig, img, ax, t)
    return fig


def fig_probabilidades(probabilidades, oscuro=True):
    """Barras con la probabilidad (%) que la red asigna a cada género."""
    fig, ax, t = _nueva_figura("Probabilidad por género", oscuro)
    ordenado = sorted(probabilidades.items(), key=lambda kv: kv[1])
    nombres = [k for k, _ in ordenado]
    valores = [v * 100 for _, v in ordenado]
    colores = [t["acento"]] * len(valores)
    colores[-1] = t["resalte"]                      # el ganador resaltado
    barras = ax.barh(nombres, valores, color=colores)
    for b, v in zip(barras, valores):
        ax.text(min(v + 1, 88), b.get_y() + b.get_height() / 2, f"{v:.1f} %",
                va="center", color=t["texto"], fontsize=8)
    ax.set_xlim(0, 100)
    _etiquetas(ax, t, "Probabilidad (%)", "")
    return fig
