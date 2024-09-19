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

from .methods.linear.transformers import (
    ClassTransformer,
    HistogramTransformer,
    DistanceTransformer,
    KernelTransformer,
    EnergyKernelTransformer,
    LaplacianKernelTransformer,
    GaussianKernelTransformer,
    GaussianRFFKernelTransformer,
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
