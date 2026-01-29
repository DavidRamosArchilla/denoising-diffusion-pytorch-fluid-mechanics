# from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion, Unet, Trainer

# from denoising_diffusion_pytorch.learned_gaussian_diffusion import LearnedGaussianDiffusion
# from denoising_diffusion_pytorch.continuous_time_gaussian_diffusion import ContinuousTimeGaussianDiffusion
# from denoising_diffusion_pytorch.weighted_objective_gaussian_diffusion import WeightedObjectiveGaussianDiffusion
# from denoising_diffusion_pytorch.elucidated_diffusion import ElucidatedDiffusion
# from denoising_diffusion_pytorch.v_param_continuous_time_gaussian_diffusion import VParamContinuousTimeGaussianDiffusion

# from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import GaussianDiffusion1D, Unet1D, Trainer1D, Dataset1D

# from denoising_diffusion_pytorch.karras_unet import (
#     KarrasUnet,
#     InvSqrtDecayLRSched
# )

# from denoising_diffusion_pytorch.karras_unet_1d import KarrasUnet1D
# from denoising_diffusion_pytorch.karras_unet_3d import KarrasUnet3D

import importlib.util
import logging
import warnings

import importlib_metadata
from packaging import version

logger = logging.getLogger(__name__)

_triton_modules_available = importlib.util.find_spec("triton") is not None
try:
    if _triton_modules_available:
        _triton_version = importlib_metadata.version("triton")
        if version.Version(_triton_version) < version.Version("3.0.0"):
            raise ValueError("triton is installed but requires Triton >= 3.0.0")
        logger.debug(f"Successfully imported triton version {_triton_version}")
except ImportError:
    _triton_modules_available = False
    warnings.warn("TritonLiteMLA and TritonMBConvPreGLU with `triton` is not available on your platform.")


def is_triton_module_available():
    return _triton_modules_available