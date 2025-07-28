import time


class OnvifPidController:
    """
    A PID controller for ONVIF camera pan, tilt, and zoom to keep a bounding box
    centered within a 4K field of view.
    """

    def __init__(
        self,
        image_width: int = 3840,
        image_height: int = 2160,
        target_bbox_width_ratio: float = 0.2,  # Desired bbox width as a ratio of image width
        kp_pan: float = 0.001,
        ki_pan: float = 0.0,
        kd_pan: float = 0.0,
        kp_tilt: float = 0.001,
        ki_tilt: float = 0.0,
        kd_tilt: float = 0.0,
        kp_zoom: float = 0.01,
        ki_zoom: float = 0.0,
        kd_zoom: float = 0.0,
        max_velocity: float = 1.0,
    ):
        """
        Initializes the PID controller.

        Args:
            image_width: The width of the image in pixels (e.g., 3840 for 4K).
            image_height: The height of the image in pixels (e.g., 2160 for 4K).
            target_bbox_width_ratio: The desired width of the bounding box as a
                                     ratio of the image width. Used for zoom control.
            kp_pan, ki_pan, kd_pan: PID gains for pan control.
            kp_tilt, ki_tilt, kd_tilt: PID gains for tilt control.
            kp_zoom, ki_zoom, kd_zoom: PID gains for zoom control.
            max_velocity: Maximum absolute velocity output for pan, tilt, and zoom.
        """
        self.image_width = image_width
        self.image_height = image_height
        self.target_bbox_width_ratio = target_bbox_width_ratio
        self.image_center_x = image_width / 2
        self.image_center_y = image_height / 2

        self.kp_pan, self.ki_pan, self.kd_pan = kp_pan, ki_pan, kd_pan
        self.kp_tilt, self.ki_tilt, self.kd_tilt = kp_tilt, ki_tilt, kd_tilt
        self.kp_zoom, self.ki_zoom, self.kd_zoom = kp_zoom, ki_zoom, kd_zoom

        self.max_velocity = max_velocity

        # PID state variables for Pan
        self._last_pan_error = 0.0
        self._integral_pan_error = 0.0
        self._last_pan_time = None

        # PID state variables for Tilt
        self._last_tilt_error = 0.0
        self._integral_tilt_error = 0.0
        self._last_tilt_time = None

        # PID state variables for Zoom
        self._last_zoom_error = 0.0
        self._integral_zoom_error = 0.0
        self._last_zoom_time = None

        self.pan_velocity = 0.0
        self.tilt_velocity = 0.0
        self.zoom_velocity = 0.0

    def _calculate_pid_output(
        self,
        current_error: float,
        last_error: float,
        integral_error: float,
        last_time: float,
        kp: float,
        ki: float,
        kd: float,
    ) -> tuple[float, float, float]:
        """
        Calculates the PID output for a single axis.

        Returns:
            A tuple containing (output, new_integral_error, new_last_error).
        """
        current_time = time.time()
        dt = current_time - last_time if last_time is not None else 0.0

        # Proportional term
        p_term = kp * current_error

        # Integral term
        new_integral_error = integral_error + current_error * dt
        i_term = ki * new_integral_error

        # Derivative term
        d_term = 0.0
        if dt > 0:
            derivative_error = (current_error - last_error) / dt
            d_term = kd * derivative_error

        output = p_term + i_term + d_term

        # Clamp output to max_velocity
        output = max(-self.max_velocity, min(self.max_velocity, output))

        return output, new_integral_error, current_error, current_time

    def update(self, bounding_box: list[int]):
        """
        Updates the PID controller with a new bounding box and calculates
        the required pan, tilt, and zoom velocities.

        Args:
            bounding_box: A list representing the bounding box in [x_min, y_min, x_max, y_max] format.
        """
        if not bounding_box or len(bounding_box) != 4:
            # If no detection or invalid bbox, stop movement
            self.pan_velocity = 0.0
            self.tilt_velocity = 0.0
            self.zoom_velocity = 0.0
            self._integral_pan_error = 0.0
            self._integral_tilt_error = 0.0
            self._integral_zoom_error = 0.0
            self._last_pan_time = None
            self._last_tilt_time = None
            self._last_zoom_time = None
            return

        x_min, y_min, x_max, y_max = bounding_box
        bbox_center_x = (x_min + x_max) / 2
        bbox_center_y = (y_min + y_max) / 2
        bbox_width = x_max - x_min

        # --- Pan Control ---
        # Error is positive if bbox is to the right of center, negative if to the left
        pan_error = bbox_center_x - self.image_center_x
        (
            self.pan_velocity,
            self._integral_pan_error,
            self._last_pan_error,
            self._last_pan_time,
        ) = self._calculate_pid_output(
            pan_error,
            self._last_pan_error,
            self._integral_pan_error,
            self._last_pan_time,
            self.kp_pan,
            self.ki_pan,
            self.kd_pan,
        )

        # --- Tilt Control ---
        # Error is positive if bbox is below center, negative if above
        tilt_error = bbox_center_y - self.image_center_y
        (
            self.tilt_velocity,
            self._integral_tilt_error,
            self._last_tilt_error,
            self._last_tilt_time,
        ) = self._calculate_pid_output(
            tilt_error,
            self._last_tilt_error,
            self._integral_tilt_error,
            self._last_tilt_time,
            self.kp_tilt,
            self.ki_tilt,
            self.kd_tilt,
        )

        # --- Zoom Control ---
        # Error is positive if bbox is too small, negative if too large
        target_bbox_width = self.image_width * self.target_bbox_width_ratio
        zoom_error = target_bbox_width - bbox_width
        (
            self.zoom_velocity,
            self._integral_zoom_error,
            self._last_zoom_error,
            self._last_zoom_time,
        ) = self._calculate_pid_output(
            zoom_error,
            self._last_zoom_error,
            self._integral_zoom_error,
            self._last_zoom_time,
            self.kp_zoom,
            self.ki_zoom,
            self.kd_zoom,
        )

    def get_velocities(self) -> tuple[float, float, float]:
        """
        Returns the calculated pan, tilt, and zoom velocities.
        These values are typically in the range [-1.0, 1.0] for ONVIF PTZ control.

        Returns:
            A tuple (pan_velocity, tilt_velocity, zoom_velocity).
        """
        return self.pan_velocity, self.tilt_velocity, self.zoom_velocity


if __name__ == "__main__":
    # Example Usage:
    print("Initializing ONVIF PID Controller...")
    controller = OnvifPidController(
        image_width=3840,
        image_height=2160,
        target_bbox_width_ratio=0.2,  # Aim for bbox to be 20% of image width
        kp_pan=0.0005,
        ki_pan=0.00001,
        kd_pan=0.0001,
        kp_tilt=0.0005,
        ki_tilt=0.00001,
        kd_tilt=0.0001,
        kp_zoom=0.005,
        ki_zoom=0.0001,
        kd_zoom=0.001,
    )

    # Simulate a bounding box moving around and changing size
    print("\\nSimulating object tracking...")
    # Initial bbox: slightly off-center, too small
    current_bbox = [1000, 800, 1500, 1200]  # x_min, y_min, x_max, y_max

    for i in range(20):
        print(f"--- Iteration {i+1} ---")
        print(f"Current Bounding Box: {current_bbox}")

        controller.update(current_bbox)
        pan_v, tilt_v, zoom_v = controller.get_velocities()

        print(
            f"Calculated Velocities: Pan={pan_v:.4f}, Tilt={tilt_v:.4f}, Zoom={zoom_v:.4f}"
        )

        # Simulate movement based on velocities (very simplified for demonstration)
        # In a real scenario, these velocities would be sent to the ONVIF camera
        # and the camera's actual position would change, affecting the next bbox.
        # Here, we just nudge the bbox towards the center/target size.
        bbox_center_x = (current_bbox[0] + current_bbox[2]) / 2
        bbox_center_y = (current_bbox[1] + current_bbox[3]) / 2
        bbox_width = current_bbox[2] - current_bbox[0]

        # Nudge bbox center towards image center
        current_bbox[0] -= int(pan_v * 50)  # Adjust multiplier for simulation speed
        current_bbox[2] -= int(pan_v * 50)
        current_bbox[1] -= int(tilt_v * 50)
        current_bbox[3] -= int(tilt_v * 50)

        # Nudge bbox width towards target width
        target_width = controller.image_width * controller.target_bbox_width_ratio
        width_diff = target_width - bbox_width
        zoom_adjust = int(zoom_v * 20)  # Adjust multiplier for simulation speed

        current_bbox[0] += zoom_adjust
        current_bbox[2] -= zoom_adjust
        current_bbox[1] += zoom_adjust * (
            controller.image_height / controller.image_width
        )  # Maintain aspect ratio
        current_bbox[3] -= zoom_adjust * (
            controller.image_height / controller.image_width
        )

        # Ensure bbox coordinates remain valid (within image bounds, x_min < x_max, etc.)
        current_bbox = [
            max(0, current_bbox[0]),
            max(0, current_bbox[1]),
            min(controller.image_width, current_bbox[2]),
            min(controller.image_height, current_bbox[3]),
        ]
        if current_bbox[0] >= current_bbox[2]:
            current_bbox[2] = current_bbox[0] + 10  # Prevent inverted bbox
        if current_bbox[1] >= current_bbox[3]:
            current_bbox[3] = current_bbox[1] + 10

        time.sleep(0.1)  # Simulate a small delay between updates

    print("\\nSimulation complete.")
    print("Final Bounding Box:", current_bbox)
    print("Final Velocities:", controller.get_velocities())

    print("\\nSimulating no detection (bbox is None or empty)...")
    controller.update([])  # Simulate no detection
    pan_v, tilt_v, zoom_v = controller.get_velocities()
    print(
        f"Calculated Velocities (no detection): Pan={pan_v:.4f}, Tilt={tilt_v:.4f}, Zoom={zoom_v:.4f}"
    )
