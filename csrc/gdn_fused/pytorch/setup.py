# SPDX-License-Identifier: Apache-2.0

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

import habana_frameworks.torch as htorch
from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir

# Link only the Habana backend library: it exports registerUserCustomOp and getUserCustomOpDescriptor.
# UserCustomOpDescriptor::execute comes from the bridge-mode plugin the interpreter has already loaded
# (eager: libhabana_pytorch2_plugin, lazy: libhabana_pytorch_plugin), so one build loads in both modes.
# A fork build of PyTorch names the library without ".upstream". See docs/design/flash-next/native-extensions.md.
BACKEND_LIBRARY = ("habana_pytorch_backend" if getattr(htorch, "is_torch_fork", False)
                   else "habana_pytorch_backend.upstream")

setup(
    name="gdn_fused_ops",
    ext_modules=[
        CppExtension(
            "gdn_fused_ops",
            ["hpu_gdn_core_decode.cpp"],
            include_dirs=[get_include_dir(), "/usr/include/habanalabs", ".."],
            library_dirs=[get_lib_dir()],
            libraries=[BACKEND_LIBRARY],
            extra_compile_args=["-O2", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
