__version__ = "0.1.5-rc4"

from .methods.linear.losses import (
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

from .methods.likelihood import (
    LikelihoodMaximizer,
    ExpectationMaximizer,
)

from .methods import AbstractMethod
