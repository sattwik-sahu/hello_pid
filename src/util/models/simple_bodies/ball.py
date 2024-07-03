from util.models import Body, Renderable, Vector
import numpy as np
from matplotlib.axes import Axes
from matplotlib import pyplot as plt


class Ball(Body, Renderable):
    def __init__(self, mass: float, radius: float) -> None:
        super().__init__(mass)
        self.radius = radius
        self.radius_ = radius
        self.patch = None

    def update(self, dt: float, force: Vector) -> None:
        self._apply_force(force=force, dt=dt)
        self.radius_ = max(1.0, self.radius_ + 0.005 * np.linalg.norm(self.acc))

    def render(self, ax: Axes) -> None:
        if self.patch is None:
            self.patch = ax.add_patch(
                plt.Circle(
                    (self.pos[0], self.pos[1]),
                    self.radius_,
                    color="red",
                    fill=False,
                    linewidth=1,
                )
            )
        self.patch.set_clip_on(False)

        # Set the patch position
        self.patch.center = (self.pos[0], self.pos[1])
