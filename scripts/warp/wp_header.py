import warp as wp
from typing import Any
import numpy as np


wp.config.mode = "release"
wp.config.enable_backward = False
wp.config.kernel_cache_dir = "./__wpcache__"
wp.config.cache_kernels = True
wp.config.verbose = False
wp.config.verbose_warnings = True
wp.config.verify_fp = False
wp.config.verify_cuda = False
wp_device = "cuda"
wp.set_device(wp_device)
wp.init()
wp.rand_init(42)
np.random.seed(42)

wp_with_torch = False
if wp_with_torch:
    import torch

    torch.set_default_device(wp_device)
    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(42)
