"""
grabador.py
===========
Captura audio del micrófono con la librería `sounddevice` (compatible con
Python 3.12; incluye PortAudio en Windows) y lo guarda como .wav con `soundfile`.

Flujo:  iniciar() -> (el audio se acumula) -> detener() -> guardar_wav()
"""
import threading
from datetime import datetime

import numpy as np
import soundfile as sf

import config


class ErrorMicrofono(Exception):
    """Problema con el micrófono (mensaje apto para el usuario)."""


class Grabador:
    def __init__(self):
        self._bloques = []
        self._flujo = None
        self._lock = threading.Lock()
        self.sr = config.SR

    @property
    def grabando(self):
        return self._flujo is not None

    def iniciar(self):
        """Abre el micrófono por defecto y comienza a capturar."""
        if self.grabando:
            return
        try:
            import sounddevice as sd
        except Exception as e:      # ImportError o falta de PortAudio
            raise ErrorMicrofono("No se pudo cargar 'sounddevice'.\n"
                                 "Ejecuta: pip install sounddevice\n" + str(e)) from e
        try:
            dispositivo = sd.query_devices(kind="input")
        except Exception as e:
            raise ErrorMicrofono(
                "No se encontró ningún dispositivo de entrada (micrófono).\n"
                "Conecta uno y revisa Windows > Configuración > Sonido > Entrada.") from e

        # Se prefiere 22050 Hz; si el dispositivo no lo admite se usa su frecuencia
        # nativa (después librosa remuestrea a 22050 Hz al procesar).
        self.sr = config.SR
        try:
            sd.check_input_settings(samplerate=self.sr, channels=1)
        except Exception:
            self.sr = int(dispositivo["default_samplerate"])

        self._bloques = []

        def _callback(indata, frames, tiempo, estado):
            with self._lock:
                self._bloques.append(indata[:, 0].copy())

        try:
            self._flujo = sd.InputStream(samplerate=self.sr, channels=1,
                                         dtype="float32", callback=_callback)
            self._flujo.start()
        except Exception as e:
            self._flujo = None
            raise ErrorMicrofono(
                "No se pudo abrir el micrófono.\n"
                "Posibles causas: sin permisos (Windows > Privacidad > Micrófono, "
                "permitir a aplicaciones de escritorio) o dispositivo en uso por "
                "otro programa.\n\nDetalle: " + str(e)) from e

    def detener(self):
        """Detiene la captura y devuelve la señal grabada (arreglo float32)."""
        if self._flujo is None:
            raise ErrorMicrofono("No hay una grabación en curso.")
        try:
            self._flujo.stop()
            self._flujo.close()
        except Exception:
            pass
        finally:
            self._flujo = None
        with self._lock:
            if not self._bloques:
                raise ErrorMicrofono("La grabación está vacía: no se capturó audio.")
            audio = np.concatenate(self._bloques)
            self._bloques = []
        if len(audio) == 0:
            raise ErrorMicrofono("La grabación está vacía: no se capturó audio.")
        return audio

    def cancelar(self):
        """Libera el micrófono sin guardar (se usa al cerrar la ventana)."""
        try:
            if self._flujo is not None:
                self._flujo.stop()
                self._flujo.close()
        except Exception:
            pass
        self._flujo = None

    def guardar_wav(self, audio):
        """Guarda la grabación en resultados/predicciones y devuelve la ruta."""
        try:
            config.DIR_PREDICCIONES.mkdir(parents=True, exist_ok=True)
            ruta = config.DIR_PREDICCIONES / f"grabacion_{datetime.now():%Y%m%d_%H%M%S}.wav"
            sf.write(str(ruta), audio, self.sr)
            return ruta
        except Exception as e:
            raise ErrorMicrofono(f"No se pudo guardar la grabación:\n{e}") from e
