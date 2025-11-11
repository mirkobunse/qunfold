__version__ = "0.1.6-rc1"

from .methods.linear.losses import (
    AbstractLoss,
    FunctionLoss,
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
    AbstractRepresentation,
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


from .methods.likelihood import (
    LikelihoodMaximizer,
    ExpectationMaximizer,
)

from .methods import AbstractMethod
