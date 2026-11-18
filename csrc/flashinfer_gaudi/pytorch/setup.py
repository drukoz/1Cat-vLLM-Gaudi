# SPDX-License-Identifier: Apache-2.0

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

import habana_frameworks.torch as htorch
from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir

# The Habana fork build of PyTorch loads libhabana_pytorch2_plugin.so; an upstream PyTorch loads the ".upstream"
# library. Link the one this interpreter loads: the other is not on the loader path, so the extension fails to load.
PLUGIN_LIBRARY = ("habana_pytorch2_plugin" if getattr(htorch, "is_torch_fork", False)
                  else "habana_pytorch2_plugin.upstream")

setup(
    name="flashinfer_gaudi_ops",
    ext_modules=[
        CppExtension(
            "flashinfer_gaudi_ops",
            ["hpu_flashinfer_gaudi.cpp", "hpu_flashinfer_activation.cpp", "hpu_flashinfer_norm.cpp"],
            include_dirs=[get_include_dir(), "/usr/include/habanalabs"],
            library_dirs=[get_lib_dir()],
            libraries=[PLUGIN_LIBRARY],
            extra_compile_args=["-O2", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
