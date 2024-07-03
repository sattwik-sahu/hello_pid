import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class Body:
    """
    Represents a physical body with mass, position, acceleration, and velocity.

    Attributes:
        mass (float): The mass of the body.
        position (np.ndarray): The position of the body in 2D space [x, y].
        acceleration (np.ndarray): The acceleration of the body in 2D space [ax, ay].
        velocity (np.ndarray): The velocity of the body in 2D space [vx, vy].
    """

    def __init__(
        self,
        mass: float,
        position: Tuple[float, float],
        acceleration: Tuple[float, float] = (0.0, 0.0),
        velocity: Tuple[float, float] = (0.0, 0.0),
    ):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.acceleration = np.array(acceleration, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

    def update(self, force: Tuple[float, float], dt: float):
        """
        Update the position, velocity, and acceleration of the body based on the applied force and time step.

        Args:
            force (Tuple[float, float]): The force applied to the body in 2D space (Fx, Fy).
            dt (float): The time step for the update.
        """
        # Calculate acceleration from Newton's second law F = ma
        ax = force[0] / self.mass
        ay = force[1] / self.mass

        # Update velocity
        self.velocity += np.array([ax * dt, ay * dt])

        # Update position
        self.position += self.velocity * dt


class Renderable(ABC):
    """
    Abstract base class for renderable objects that can be rendered using matplotlib.
    """

    @abstractmethod
    def render(self, ax: plt.Axes):
        """
        Abstract method to render the object using matplotlib.

        Args:
            ax (matplotlib.axes.Axes): The axes object to render the object on.
        """
        pass


class Ball(Body, Renderable):
    """
    Represents a ball, inheriting properties of both Body and Renderable.

    Attributes:
        radius (float): The radius of the ball.
        color (str): The color of the ball for rendering.
    """

    def __init__(
        self,
        mass: float,
        position: Tuple[float, float],
        radius: float,
        acceleration: Tuple[float, float] = (0.0, 0.0),
        velocity: Tuple[float, float] = (0.0, 0.0),
        color: str = "blue",
    ):
        super().__init__(mass, position, acceleration, velocity)
        self.radius = radius
        self.color = color

    def render(self, ax: plt.Axes):
        """
        Render the ball using matplotlib.

        Args:
            ax (matplotlib.axes.Axes): The axes object to render the ball on.
        """
        circle = plt.Circle(self.position, self.radius, color=self.color)
        ax.add_artist(circle)


class Marker(Renderable):
    """
    Represents a marker at a specific position.

    Attributes:
        color (str): The color of the marker.
        marker_style (str): The style of the marker (default is 'x').
    """

    def __init__(
        self,
        color: str = "green",
        marker_style: str = "x",
    ):
        self.color = color
        self.marker_style = marker_style
        self.position = np.zeros(2, dtype=float)  # Start at the origin

    def update_position(self):
        """
        Update the position of the marker randomly.
        """
        # Update position in a random walk manner
        self.position = self.position * (1 - 1e-6) + 10000.0 * np.random.uniform(-10.0, 10.0, size=2) * 1e-6

    def render(self, ax: plt.Axes):
        """
        Render the marker using matplotlib.

        Args:
            ax (matplotlib.axes.Axes): The axes object to render the marker on.
        """
        ax.plot(
            self.position[0],
            self.position[1],
            marker=self.marker_style,
            color=self.color,
        )


class PIDControl(ABC):
    """
    Abstract base class for a PID controller.

    Attributes:
        target_position (np.ndarray): Target position to move towards.
    """

    def __init__(self, target_position: Tuple[float, float]):
        self.target_position = np.array(target_position, dtype=float)

    @abstractmethod
    def update(
        self, current_position: Tuple[float, float], dt: float
    ) -> Tuple[float, float]:
        """
        Abstract method to update the PID controller and calculate the control output.

        Args:
            current_position (Tuple[float, float]): Current position of the controlled object.
            dt (float): Time step for the update.

        Returns:
            Tuple[float, float]: The control output (force) to be applied.
        """
        pass


class SimplePIDControl(PIDControl):
    """
    Simple PID controller implementation for controlling the position of an object.

    Attributes:
        kp (float): Proportional gain.
        ki (float): Integral gain.
        kd (float): Derivative gain.
        target_position (Tuple[float, float]): Target position to move towards.
        current_position (np.ndarray): Current position of the object.
        integral_error (np.ndarray): Integral error for accumulating errors over time.
        prev_error (np.ndarray): Previous error for calculating derivative term.
    """

    def __init__(
        self, kp: float, ki: float, kd: float, target_position: Tuple[float, float]
    ):
        super().__init__(target_position)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.current_position = np.array(
            target_position, dtype=float
        )  # Start at the target position
        self.integral_error = np.zeros(2, dtype=float)
        self.prev_error = np.zeros(2, dtype=float)

    def update_target_position(self, target_position: Tuple[float, float]):
        """
        Update the target position of the controller.

        Args:
            target_position (Tuple[float, float]): The new target position.
        """
        self.target_position = np.array(target_position, dtype=float)
 
    def update(
        self, current_position: Tuple[float, float], dt: float
    ) -> Tuple[float, float]:
        """
        Update the PID controller and calculate the control output (force).

        Args:
            current_position (Tuple[float, float]): Current position of the object.
            dt (float): Time step for the update.

        Returns:
            Tuple[float, float]: The control output (force) to be applied to the object.
        """
        self.current_position = np.array(current_position, dtype=float)

        # Calculate error terms
        error = self.target_position - self.current_position
        self.integral_error += error * dt
        derivative_error = (error - self.prev_error) / dt

        # PID control output (force)
        force = (
            self.kp * error + self.ki * self.integral_error + self.kd * derivative_error
        )

        # Update previous error for next iteration
        self.prev_error = error

        return tuple(force)


# Animation function
def animate(frame):
    global ball, pid_controller, dt, marker

    # Update marker position randomly
    marker.update_position()

    # Update PID controller and get force
    pid_controller.update_target_position(marker.position)
    force = pid_controller.update(ball.position, dt)

    # Update ball position
    ball.update(force, dt)

    # Clear previous frame
    ax.clear()

    # Render the ball and marker
    ball.render(ax)
    marker.render(ax)

    # Set plot limits
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal", adjustable="box")

    # Set title
    ax.set_title("PID Controlled Ball Movement")

    return (ax,)


# Main function
if __name__ == "__main__":
    # Create a ball and PID controller
    ball = Ball(mass=10.0, position=(0.0, 0.0), radius=0.2, color="red")
    pid_controller = SimplePIDControl(
        kp=400.0, ki=100.0, kd=200.0, target_position=(4.0, 3.0)
    )

    # Create a marker
    marker = Marker(color="green")

    # Animation setup
    fig, ax = plt.subplots()
    dt = 0.02  # Time step for simulation

    ani = animation.FuncAnimation(fig, animate, frames=200, interval=20, blit=True)

    # Show the animation
    plt.show()
