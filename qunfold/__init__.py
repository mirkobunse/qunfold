__version__ = "0.1.5-rc10"

from .methods.linear.losses import (
    LeastSquaresLoss,
    BlobelLoss,
    EnergyLoss,
    HellingerSurrogateLoss,
    CombinedLoss,
    TikhonovRegularization,
    TikhonovRegularized,
    KDEyHDLoss,
    KDEyCSLoss,
    KDEyMLLoss,
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
    OriginalRepresentation,
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

from .methods.kernel_density import (
    KDEyML,
    KDEyMLQP,
)

from .methods.likelihood import (
    LikelihoodMaximizer,
    ExpectationMaximizer,
)

from .methods import AbstractMethod
