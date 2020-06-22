import shutil
import subprocess
import tempfile
from pathlib import Path

from det3d.utils.find import find_cuda_device_arch
from det3d.utils.loader import import_file

from .command import CUDALink, Gpp, Nvcc, compile_libraries, out


class Pybind11Link(Gpp):
    def __init__(
        self,
        sources,
        target,
        std="c++11",
        includes: list = None,
        defines: dict = None,
        cflags: str = None,
        libraries: dict = None,
        lflags: str = None,
        extra_cflags: str = None,
        extra_lflags: str = None,
        build_directory: str = None,
    ):
        pb11_includes = (
            subprocess.check_output("python3 -m pybind11 --includes", shell=True)
            .decode("utf8")
            .strip("\n")
        )
        cflags = cflags or "-fPIC -O3 "
        cflags += pb11_includes
        super().__init__(
            sources,
            target,
            std,
            includes,
            defines,
            cflags,
            link=True,
            libraries=libraries,
            lflags=lflags,
            extra_cflags=extra_cflags,
            extra_lflags=extra_lflags,
            build_directory=build_directory,
        )


class Pybind11CUDALink(CUDALink):
    def __init__(
        self,
        sources,
        target,
        std="c++11",
        includes: list = None,
        defines: dict = None,
        cflags: str = None,
        libraries: dict = None,
        lflags: str = None,
        extra_cflags: str = None,
        extra_lflags: str = None,
        build_directory: str = None,
    ):
        pb11_includes = (
            subprocess.check_output("python3 -m pybind11 --includes", shell=True)
            .decode("utf8")
            .strip("\n")
        )
        cflags = cflags or "-fPIC -O3 "
        cflags += pb11_includes
        super().__init__(
            sources,
            target,
            std,
            includes,
            defines,
            cflags,
            libraries=libraries,
            lflags=lflags,
            extra_cflags=extra_cflags,
            extra_lflags=extra_lflags,
            build_directory=build_directory,
        )


def load_pb11(
    sources,
    target,
    cwd=".",
    cuda=False,
    arch=None,
    num_workers=4,
    includes: list = None,
    build_directory=None,
    compiler="g++",
):
    cmd_groups = []
    cmds = []
    outs = []
    main_sources = []
    if arch is None:
        arch = find_cuda_device_arch()

    for s in sources:
        s = str(s)
        if ".cu" in s or ".cu.cc" in s:
            assert cuda is True, "cuda must be true if contain cuda file"
            cmds.append(Nvcc(s, out(s), arch))
            outs.append(out(s))
        else:
            main_sources.append(s)

    if cuda is True and arch is None:
        raise ValueError("you must specify arch if sources contains" " cuda files")
    cmd_groups.append(cmds)
    if cuda:
        cmd_groups.append(
            [Pybind11CUDALink(outs + main_sources, target, includes=includes)]
        )
    else:
        cmd_groups.append(
            [Pybind11Link(outs + main_sources, target, includes=includes)]
        )
    for cmds in cmd_groups:
        compile_libraries(cmds, cwd, num_workers=num_workers, compiler=compiler)

    return import_file(target, add_to_sys=False, disable_warning=True)
