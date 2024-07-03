from matplotlib import pyplot as plt


class Plotter:
    def __init__(self, fig: plt.Figure, ax: plt.Axes) -> None:
        self.fig = fig
        self.ax = ax

        self.patches = []
        self.lines = []
        self.texts = []
        self.artists = self.patches + self.lines + self.texts

        self.ax.set_aspect("equal")

    def add_patch(self, patch: plt.Patch) -> None:
        self.patches.append(patch)
        self.artists.append(patch)
        self.ax.add_patch(patch)

    def show(self) -> None:
        self.fig.show()
