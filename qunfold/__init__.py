from .losses import (
    instantiate_loss,
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

from .transformers import (
    ClassTransformer,
    HistogramTransformer,
    DistanceTransformer,
    KernelTransformer,
    EnergyKernelTransformer,
    LaplacianKernelTransformer,
    GaussianKernelTransformer,
    GaussianRFFKernelTransformer,
    KDEyHDTransformer,
    KDEyCSTransformer,
    KDEyMLTransformerID,
)

from .methods.linear import (
    AbstractMethod,
    LinearMethod,
    ACC,
    PACC,
    RUN,
    HDx,
    HDy,
    EDx,
    EDy,
    KMM,
    KDEyHD,
    KDEyCS,
    KDEyMLID,
)

from .methods.kernel_density import (
    KDEyML,
)
