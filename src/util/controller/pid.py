class PIDController:
    """
    A Proportional-Integral-Derivative (PID) controller class.
    
    Attributes:
        k_p (float): Proportional gain.
        k_i (float): Integral gain.
        k_d (float): Derivative gain.
        output (float): Current output of the PID controller.
        target (float): Target setpoint for the controller.
        error_integral (float): Integral of the error over time.
        prev_error (float): Previous error value for derivative calculation.
    """

    def __init__(self, k_p: float, k_i: float, k_d: float) -> None:
        """
        Initialize PIDController with specified gains.

        Args:
            k_p (float): Proportional gain.
            k_i (float): Integral gain.
            k_d (float): Derivative gain.
        """
        self.output: float = 0.0
        self.target: float = 0.0

        self.k_p: float = k_p
        self.k_i: float = k_i
        self.k_d: float = k_d

        self.error_integral: float = 0.0
        self.prev_error: float = 0.0

    def set_target(self, target: float) -> None:
        """
        Set the target setpoint for the PID controller.

        Args:
            target (float): Target setpoint.
        """
        self.target = target

    def _calc_error_integral(self, error: float, dt: float) -> float:
        """
        Calculate the integral of the error over time.

        Args:
            error (float): Current error value.
            dt (float): Time step duration.

        Returns:
            float: Integral of the error.
        """
        self.error_integral += error * dt
        return self.error_integral

    def _calc_error_derivative(self, error: float, dt: float) -> float:
        """
        Calculate the derivative of the error with respect to time.

        Args:
            error (float): Current error value.
            dt (float): Time step duration.

        Returns:
            float: Derivative of the error.
        """
        error_derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return error_derivative

    def update(self, current_value: float, dt: float) -> float:
        """
        Update the PID controller output based on current value and time step.

        Args:
            current_value (float): Current value of the process being controlled.
            dt (float): Time step duration.

        Returns:
            float: Output of the PID controller.
        """
        error = self.target - current_value
        self.output = (
            self.k_p * error
            + self.k_i * self._calc_error_integral(error=error, dt=dt)
            + self.k_d * self._calc_error_derivative(error=error, dt=dt)
        )
        return self.output
