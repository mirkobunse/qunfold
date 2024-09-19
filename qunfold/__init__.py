__version__ = "0.1.5-rc"

from .methods.linear.losses import (
    instantiate_loss,
    LeastSquaresLoss,
    BlobelLoss,
    EnergyLoss,
    HellingerSurrogateLoss,
    CombinedLoss,
    TikhonovRegularization,
    TikhonovRegularized,
)

from .methods.linear.representations import (
    ClassRepresentation,
    HistogramRepresentation,
    DistanceRepresentation,
    KernelRepresentation,
    EnergyKernelRepresentation,
    LaplacianKernelRepresentation,
    GaussianKernelRepresentation,
    GaussianRFFKernelRepresentation,
)

from .methods.linear import (
    LinearMethod,
    ACC,
    PACC,
    RUN,
    HDx,
    HDy,
    EDx,
    EDy,
    KMM,
)

from .methods import AbstractMethod
