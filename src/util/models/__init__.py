from typing import Tuple
import numpy as np
from abc import ABC, abstractmethod

Vector = np.ndarray


class Body(ABC):
    pos: Vector
    vel: Vector
    acc: Vector

    def __init__(self, mass: float) -> None:
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self.acc = np.zeros(2)

        self.mass = mass

    def __repr__(self) -> str:
        return f"Body(pos={self.pos}, vel={self.vel}, acc={self.acc})"

    def __str__(self) -> str:
        return f"Body(pos={self.pos}, vel={self.vel}, acc={self.acc})"

    def _apply_force(self, force: Vector, dt: float) -> None:
        self.acc += force / self.mass
        self.vel += self.acc * dt
        self.pos += self.vel * dt

    @abstractmethod
    def update(self, dt: float, *args, **kwargs) -> None:
        pass


class Renderable(ABC):
    @abstractmethod
    def render(self, *args, **kwargs) -> None:
        pass
