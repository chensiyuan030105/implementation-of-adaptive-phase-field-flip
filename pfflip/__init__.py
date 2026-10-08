"""PF-FLIP: Phase-Field-FLIP two-phase fluid simulation.

Reproduction of Braun, Bender & Thuerey, "Adaptive Phase-Field-FLIP for Very
Large Scale Two-Phase Fluid Simulation", ACM TOG 44(4), SIGGRAPH 2025.
"""

from .config import Config
from .sim import Simulator

__all__ = ["Config", "Simulator", "scenes"]
