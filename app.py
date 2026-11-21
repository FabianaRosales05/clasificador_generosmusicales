
import queue
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import config
import visualizacion as vz
from extraer_caracteristicas import ErrorAudio
from grabador import ErrorMicrofono, Grabador
from predecir import ClasificadorGeneros, ErrorModelo

# PALETA RASPBERRY PINK

# Fondo principal
BG = "#140A11"

# Tarjetas y paneles
PANEL = "#2D1826"

# Texto principal
TEXTO = "#FFF6FB"

# Texto secundario
SUAVE = "#F1A7CB"

# Botones principales
ACENTO = "#E25098"

# Hover de botones
ACENTO_H = "#DB247E"

# Color de éxito
VERDE = "#E25098"

# Grabación y errores
ROJO = "#AF1D65"
ROJO_H = "#DB247E"

NOMBRES_MODELO = {"MLP (características)": "mlp", "CNN (Mel-espectrograma)": "cnn"}

class Aplicacion:
    def __init__(self, raiz):
        self.raiz = raiz
        raiz.title("Clasificador de Géneros Musicales")
        raiz.geometry("1280x800")
        raiz.minsize(1100, 720)
        raiz.configure(bg=BG)

        self.cola = queue.Queue()
        self.clasificadores = {}
        self.cargando = threading.Lock()
        self.grabador = Grabador()
        self.ocupado = False
        self.inicio_grab = 0.0
        self.duracion_grab = config.MIC_DURACION_DEFECTO
        self.id_timer = None

        self._estilo()
        self._construir()
        self.raiz.protocol("WM_DELETE_WINDOW", self._cerrar)
        self.raiz.after(100, self._revisar_cola)

        if not ClasificadorGeneros.modelos_disponibles():
            self._estado("Aún no hay modelo entrenado: ejecuta  python entrenar_modelo.py",
                         ROJO_H)

    # ------------------------------------------------------------ interfaz
    def _estilo(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("Conf.Horizontal.TProgressbar", troughcolor=BG, background=VERDE,
                    bordercolor=PANEL, lightcolor=VERDE, darkcolor=VERDE)
        s.configure("TCombobox", fieldbackground=PANEL, background=PANEL,
                    foreground=TEXTO, arrowcolor=TEXTO)

    def _boton(self, padre, texto, comando, color, hover):
        b = tk.Button(padre, text=texto, command=comando, bg=color, fg="white",
                      activebackground=hover, activeforeground="white",
                      font=("Segoe UI", 12, "bold"), relief="flat", bd=0,
                      padx=26, pady=14, cursor="hand2")
        b.bind("<Enter>", lambda e: b["state"] == "normal" and b.config(bg=hover))
        b.bind("<Leave>", lambda e: b.config(bg=color))
        return b

    def _construir(self):
        # ---- Título
        cab = tk.Frame(self.raiz, bg=BG)
        cab.pack(fill="x", pady=(18, 4))
        tk.Label(cab, text="CLASIFICADOR DE GÉNEROS MUSICALES", bg=BG, fg=TEXTO,
                 font=("Segoe UI", 24, "bold")).pack()
        tk.Label(cab, text="Redes Neuronales  ·  Procesamiento de audio  ·  MLP / CNN",
                 bg=BG, fg=SUAVE, font=("Segoe UI", 10)).pack()

        # ---- Controles
        ctr = tk.Frame(self.raiz, bg=BG)
        ctr.pack(pady=14)
        self.btn_cargar = self._boton(ctr, "📂  CARGAR AUDIO", self.cargar_audio,
                                      ACENTO, ACENTO_H)
        self.btn_cargar.grid(row=0, column=0, padx=10)
        self.btn_grabar = self._boton(ctr, "🎙  GRABAR CON MICRÓFONO", self.alternar_grabacion,
                                      ACENTO, ACENTO_H)
        self.btn_grabar.grid(row=0, column=1, padx=10)

        opc = tk.Frame(ctr, bg=BG)
        opc.grid(row=0, column=2, padx=20)
        tk.Label(opc, text="Modelo", bg=BG, fg=SUAVE, font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w")
        disponibles = ClasificadorGeneros.modelos_disponibles()
        valores = [n for n, t in NOMBRES_MODELO.items() if t in disponibles] or list(NOMBRES_MODELO)
        self.var_modelo = tk.StringVar(value=valores[0])
        ttk.Combobox(opc, textvariable=self.var_modelo, values=valores, state="readonly",
                     width=24).grid(row=1, column=0, pady=(0, 4))
        tk.Label(opc, text="Duración de grabación (s)", bg=BG, fg=SUAVE,
                 font=("Segoe UI", 9)).grid(row=2, column=0, sticky="w")
        self.var_dur = tk.IntVar(value=config.MIC_DURACION_DEFECTO)
        tk.Spinbox(opc, from_=config.MIC_DURACION_MIN, to=config.MIC_DURACION_MAX,
                   textvariable=self.var_dur, width=6, bg=PANEL, fg=TEXTO,
                   buttonbackground=PANEL, relief="flat").grid(row=3, column=0, sticky="w")

        # ---- Estado
        self.lbl_estado = tk.Label(self.raiz, text="Listo. Carga un audio o graba con el micrófono.",
                                   bg=BG, fg=SUAVE, font=("Segoe UI", 11))
        self.lbl_estado.pack(pady=(0, 8))

        # ---- Cuerpo: tarjeta de resultado + gráficas
        cuerpo = tk.Frame(self.raiz, bg=BG)
        cuerpo.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        cuerpo.columnconfigure(1, weight=1)
        cuerpo.rowconfigure(0, weight=1)

        tarjeta = tk.Frame(cuerpo, bg=PANEL, width=290)
        tarjeta.grid(row=0, column=0, sticky="ns", padx=(0, 14))
        tarjeta.pack_propagate(False)
        tk.Label(tarjeta, text="GÉNERO DETECTADO", bg=PANEL, fg=SUAVE,
                 font=("Segoe UI", 11, "bold")).pack(pady=(30, 8))
        self.lbl_genero = tk.Label(tarjeta, text="—", bg=PANEL, fg=TEXTO,
                                   font=("Segoe UI", 34, "bold"))
        self.lbl_genero.pack(pady=(0, 20))
        tk.Label(tarjeta, text="Confianza", bg=PANEL, fg=SUAVE,
                 font=("Segoe UI", 11)).pack()
        self.lbl_conf = tk.Label(tarjeta, text="—", bg=PANEL, fg=VERDE,
                                 font=("Segoe UI", 28, "bold"))
        self.lbl_conf.pack(pady=(0, 10))
        self.barra = ttk.Progressbar(tarjeta, style="Conf.Horizontal.TProgressbar",
                                     maximum=100, length=220)
        self.barra.pack(pady=(0, 24))
        self.lbl_info = tk.Label(tarjeta, text="", bg=PANEL, fg=SUAVE, justify="left",
                                 wraplength=250, font=("Segoe UI", 9))
        self.lbl_info.pack(padx=18, anchor="w")

        graficas = tk.Frame(cuerpo, bg=BG)
        graficas.grid(row=0, column=1, sticky="nsew")
        for i in range(2):
            graficas.columnconfigure(i, weight=1, uniform="g")
            graficas.rowconfigure(i, weight=1, uniform="g")
        self.marcos = {}
        for clave, (f, c), texto in (("onda", (0, 0), "Forma de onda"),
                                     ("espec", (0, 1), "Espectrograma"),
                                     ("mel", (1, 0), "Mel-espectrograma"),
                                     ("prob", (1, 1), "Probabilidad por género")):
            m = tk.Frame(graficas, bg=PANEL)
            m.grid(row=f, column=c, sticky="nsew", padx=5, pady=5)
            tk.Label(m, text=texto + "\n(aparecerá tras analizar un audio)", bg=PANEL,
                     fg=SUAVE, font=("Segoe UI", 10)).pack(expand=True)
            self.marcos[clave] = m

    # ------------------------------------------------------------ utilidades UI
    def _estado(self, texto, color=SUAVE):
        self.lbl_estado.config(text=texto, fg=color)

    def _bloquear(self, bloquear):
        self.ocupado = bloquear
        estado = "disabled" if bloquear else "normal"
        self.btn_cargar.config(state=estado)

    def _mostrar_figura(self, clave, fig):
        marco = self.marcos[clave]
        for w in marco.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=marco)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------------ cargar archivo
    def cargar_audio(self):
        if self.ocupado:
            return
        ruta = filedialog.askopenfilename(
            title="Selecciona un archivo de audio",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.au"), ("Todos", "*.*")])
        if not ruta:
            return
        self._estado(f"Archivo recibido: {Path(ruta).name}. Procesando...", ACENTO_H)
        self._analizar(ruta)

    # ------------------------------------------------------------ micrófono
    def alternar_grabacion(self):
        if self.grabador.grabando:
            self._finalizar_grabacion()
        elif not self.ocupado:
            self._iniciar_grabacion()

    def _iniciar_grabacion(self):
        try:
            self.duracion_grab = int(self.var_dur.get())
        except (tk.TclError, ValueError):
            self.duracion_grab = config.MIC_DURACION_DEFECTO
        self.duracion_grab = max(config.MIC_DURACION_MIN,
                                 min(config.MIC_DURACION_MAX, self.duracion_grab))
        try:
            self.grabador.iniciar()
        except ErrorMicrofono as e:
            self._estado("No se pudo iniciar la grabación.", ROJO_H)
            messagebox.showerror("Micrófono", str(e))
            return
        import time
        self.inicio_grab = time.time()
        self.btn_grabar.config(text="■  DETENER GRABACIÓN", bg=ROJO)
        self.btn_grabar.bind("<Enter>", lambda e: self.btn_grabar.config(bg=ROJO_H))
        self.btn_grabar.bind("<Leave>", lambda e: self.btn_grabar.config(bg=ROJO))
        self.btn_cargar.config(state="disabled")
        self._tic_grabacion()

    def _tic_grabacion(self):
        if not self.grabador.grabando:
            return
        import time
        t = time.time() - self.inicio_grab
        self._estado(f"●  GRABANDO...  {t:0.1f} s / {self.duracion_grab} s", ROJO_H)
        if t >= self.duracion_grab:
            self._finalizar_grabacion()
        else:
            self.id_timer = self.raiz.after(100, self._tic_grabacion)

    def _restaurar_boton_grabar(self):
        self.btn_grabar.config(text="🎙  GRABAR CON MICRÓFONO", bg=ACENTO)
        self.btn_grabar.bind("<Enter>", lambda e: self.btn_grabar.config(bg=ACENTO_H))
        self.btn_grabar.bind("<Leave>", lambda e: self.btn_grabar.config(bg=ACENTO))

    def _finalizar_grabacion(self):
        if self.id_timer:
            self.raiz.after_cancel(self.id_timer)
            self.id_timer = None
        self._restaurar_boton_grabar()
        self.btn_cargar.config(state="normal")
        try:
            audio = self.grabador.detener()
            ruta = self.grabador.guardar_wav(audio)
        except ErrorMicrofono as e:
            self._estado("Error en la grabación.", ROJO_H)
            messagebox.showerror("Micrófono", str(e))
            return
        self._estado(f"Grabación guardada ({len(audio) / self.grabador.sr:.1f} s). "
                     "Procesando...", ACENTO_H)
        self._analizar(str(ruta))

    # ------------------------------------------------------------ análisis (hilo)
    def _analizar(self, ruta):
        tipo = NOMBRES_MODELO.get(self.var_modelo.get(), "mlp")
        self._bloquear(True)
        self.btn_grabar.config(state="disabled")
        threading.Thread(target=self._trabajo, args=(ruta, tipo), daemon=True).start()

    def _trabajo(self, ruta, tipo):
        try:
            with self.cargando:
                if tipo not in self.clasificadores:
                    self.cola.put(("estado", "Cargando modelo entrenado..."))
                    self.clasificadores[tipo] = ClasificadorGeneros(tipo)
            self.cola.put(("estado", "Extrayendo características y prediciendo..."))
            resultado = self.clasificadores[tipo].predecir_archivo(ruta)
            self.cola.put(("ok", resultado, ruta))
        except (ErrorAudio, ErrorModelo) as e:
            self.cola.put(("error", str(e)))
        except Exception as e:
            traceback.print_exc()
            self.cola.put(("error", f"Error inesperado durante el procesamiento:\n{e}"))

    def _revisar_cola(self):
        try:
            while True:
                msg = self.cola.get_nowait()
                if msg[0] == "estado":
                    self._estado(msg[1], ACENTO_H)
                elif msg[0] == "ok":
                    self._mostrar_resultado(msg[1], msg[2])
                    self._fin_analisis()
                elif msg[0] == "error":
                    self._estado("No se pudo analizar el audio.", ROJO_H)
                    messagebox.showerror("Error", msg[1])
                    self._fin_analisis()
        except queue.Empty:
            pass
        self.raiz.after(100, self._revisar_cola)

    def _fin_analisis(self):
        self._bloquear(False)
        self.btn_grabar.config(state="normal")

    def _mostrar_resultado(self, r, ruta):
        self.lbl_genero.config(text=r["genero"].upper())
        self.lbl_conf.config(text=f"{r['confianza']:.2f} %")
        self.barra["value"] = r["confianza"]
        self.lbl_info.config(text=(f"Archivo: {Path(ruta).name}\n"
                                   f"Duración analizada: {r['duracion']:.1f} s\n"
                                   f"Segmentos de {config.SEGMENTO_SEG:.0f} s: {r['n_segmentos']}\n"
                                   f"Modelo: {r['modelo']}"))
        self._mostrar_figura("onda", vz.fig_forma_onda(r["audio"], r["sr"]))
        self._mostrar_figura("espec", vz.fig_espectrograma(r["audio"], r["sr"]))
        self._mostrar_figura("mel", vz.fig_mel(r["audio"], r["sr"]))
        self._mostrar_figura("prob", vz.fig_probabilidades(r["probabilidades"]))
        self._estado(f"Análisis completado: {r['genero'].upper()} ({r['confianza']:.2f} %)", VERDE)

    def _cerrar(self):
        self.grabador.cancelar()
        self.raiz.destroy()


def main():
    raiz = tk.Tk()
    Aplicacion(raiz)
    raiz.mainloop()


if __name__ == "__main__":
    main()
