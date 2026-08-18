import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#UPDATEAR PARA QUE MUESTRE MULTIPLES LINEAS DE COLORES DISTINTOS

def create_live_loss_plot():
    """
    Crea y devuelve un objeto LiveLossPlot listo para usar.
    Llama a .update(step, loss) cada vez que tengas un nuevo dato.
    """
    return LiveLossPlot()

class linea_class:
    def __init__ (self, ax, nombre, color, width):
        self.steps = []
        self.losses = []
        (self.line,) = ax.plot([], [], color=color, linewidth=width, label=nombre)


class LiveLossPlot:
    def __init__(self, title="Training loss",x_lab = "Step", y_lab = "Loss", max_points=None):
        self.lineas =[]
        self.max_points = max_points  # None = sin límite

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.fig.canvas.manager.set_window_title(title)
        self.add_linea("0",("#000000"), 1.0)#LINEA 0, NO USADA
        self.ax.set_xlabel(x_lab)
        self.ax.set_ylabel(y_lab)
        self.ax.set_title(title)
        self.ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        self.ax.legend()
        self.ax.grid(True, linestyle="--", alpha=0.4)
        self.fig.tight_layout()
        
    def add_linea(self,nombre,color,width = 1.0):
        line = linea_class(self.ax,nombre,color,width)
        self.lineas.append(line)
        return line

    def update(self, linea, step, loss, nombre, color, refresh=True):

        """Agrega un punto y refresca el gráfico."""
        try:
            line = self.lineas[linea]#  #
            linea_nueva = False
        except:
            line = self.add_linea(nombre, color)
            linea_nueva = True

        line.steps.append(step)
        line.losses.append(loss)
        if self.max_points and len(line.steps) > self.max_points:
            line.steps = line.steps[-self.max_points :]
            line.losses = line.losses[-self.max_points :]
        #print (f"step{step}")
        
        if refresh:
            line.line.set_data(line.steps, line.losses)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        return linea_nueva


    def close(self):
        """Congela el gráfico al terminar el entrenamiento."""
        plt.ioff()
        plt.show()


# ── Ejemplo de uso ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time, math, random

    plot = create_live_loss_plot()

    for step in range(1, 10100):
        loss = 1 / math.log(step + 1) + random.uniform(-0.02, 0.02)
        plot.update(0, step, loss, "loss_test", ("#00B400"),refresh=(step % 10 == 1))
        #time.sleep(5)   # simulación de cada paso de entrenamiento

    plot.close()
