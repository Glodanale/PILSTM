from enum import Enum

class Variation(str, Enum):
    TRUNC_LINEAR = "trunc_linear"
    TRUNC_NONLINEAR = "trunc_nonlinear"
    MASK_LINEAR = "mask_linear"
    MASK_NONLINEAR = "mask_nonlinear"
