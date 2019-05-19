"""
Constants used in Maggy: Allowed datatypes etc.
"""
import numpy as np

class USER_FCT:
    """Return datatypes allowed for user defined training function.
    """
    RETURN_TYPES = (float, int, np.number)