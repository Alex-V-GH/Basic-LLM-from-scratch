import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def create_live_loss_plot():
    """
    Crea y devuelve un objeto LiveLossPlot listo para usar.
    Llama a .update(step, loss) cada vez que tengas un nuevo dato.
    """
    return LiveLossPlot()


class LiveLossPlot:
    def __init__(self, title="Training loss", max_points=None):
        self.steps = []
        self.losses = []
        self.steps2 = []
        self.losses2 = []
        self.max_points = max_points  # None = sin límite

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.fig.canvas.manager.set_window_title(title)

        (self.line,) = self.ax.plot([], [], color="#5563DE", linewidth=1.0, label="Actual_loss")
        (self.line2,) = self.ax.plot([], [], color="#DE6355", linewidth=1.0, label="Saved_loss")
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Loss")
        self.ax.set_title(title)
        self.ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        self.ax.legend()
        self.ax.grid(True, linestyle="--", alpha=0.4)
        self.fig.tight_layout()

    def update(self,linea, step: int, loss: float):
        """Agrega un punto y refresca el gráfico."""
        if linea==1:
            self.steps.append(step)
            self.losses.append(loss)

            if self.max_points and len(self.steps) > self.max_points:
                self.steps = self.steps[-self.max_points :]
                self.losses = self.losses[-self.max_points :]

            self.line.set_data(self.steps, self.losses)
        elif linea==2:
            self.steps2.append(step)
            self.losses2.append(loss)

            if self.max_points and len(self.steps) > self.max_points:
                self.steps2 = self.steps2[-self.max_points :]
                self.losses2 = self.losses2[-self.max_points :]

            self.line2.set_data(self.steps2, self.losses2)

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """Congela el gráfico al terminar el entrenamiento."""
        plt.ioff()
        plt.show()


# ── Ejemplo de uso ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time, math, random

    plot = create_live_loss_plot()

    for step in range(1, 101):
        loss = 1 / math.log(step + 1) + random.uniform(-0.02, 0.02)
        plot.update(step, loss)
        time.sleep(5)   # simulación de cada paso de entrenamiento

    plot.close()
