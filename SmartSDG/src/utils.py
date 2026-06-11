# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
# This file contains reusable helper functions for the project.

import math
import random
from typing import Tuple

def generate_orbital_camera_position(orbit_center: Tuple[float, float, float], radius: float, height: float) -> Tuple[float, float, float]:
    """
    Calculates a random camera position on a circular path around a given center.
    """
    angle = random.uniform(0, 2 * math.pi)
    cam_x = orbit_center[0] + radius * math.cos(angle)
    cam_y = orbit_center[1] + radius * math.sin(angle)
    cam_z = height
    return (cam_x, cam_y, cam_z)
