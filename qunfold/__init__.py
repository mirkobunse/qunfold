from .losses import (
    instantiate_loss,
    FunctionLoss,
    LeastSquaresLoss,
    BlobelLoss,
    EnergyLoss,
    HellingerSurrogateLoss,
    CombinedLoss,
    TikhonovRegularization,
    TikhonovRegularized,
)

from .transformers import (
    ClassTransformer,
    HistogramTransformer,
    DistanceTransformer,
    KernelTransformer,
    EnergyKernelTransformer,
    LaplacianKernelTransformer,
    GaussianKernelTransformer,
    GaussianRFFKernelTransformer,
)

from .methods import (
    GenericMethod,
    ACC,
    PACC,
    RUN,
    HDx,
    HDy,
    EDx,
    EDy,
    KMM,
)
