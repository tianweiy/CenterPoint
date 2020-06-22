import multiprocessing
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from functools import partial
from pathlib import Path

import fire
from det3d.utils.find import find_cuda, find_cuda_device_arch


class Gpp:
    def __init__(
        self,
        sources,
        target,
        std="c++11",
        includes: list = None,
        defines: dict = None,
        cflags: str = None,
        compiler="g++",
        link=False,
        libraries: dict = None,
        lflags: str = None,
        extra_cflags: str = None,
        extra_lflags: str = None,
        build_directory: str = None,
    ):
        if not isinstance(sources, (list, tuple)):
            sources = [sources]
        if build_directory is not None:
            build_directory = Path(build_directory)
            new_sources = []
            for p in sources:
                if not Path(p).is_absolute():
                    new_sources.append(str(build_directory / p))
                else:
                    new_sources.append(p)
            sources = new_sources
            target = Path(target)
            if not target.is_absolute():
                target = str(build_directory / target)
        self.sources = [str(p) for p in sources]
        self.target = str(target)
        self.std = std
        self.includes = includes or []
        self.cflags = cflags or "-fPIC -O3"
        self.defines = defines or {}
        self.compiler = compiler
        self.link = link
        self.libraries = libraries or {}
        self.lflags = lflags or ""
        self.extra_cflags = extra_cflags or ""
        self.extra_lflags = extra_lflags or ""

    def shell(self, target: str = None, compiler: str = None):
        defines = [f"-D {n}={v}" for n, v in self.defines.items()]
        includes = [f"-I{inc}" for inc in self.includes]
        libraries = [
            f"-L{n} {' '.join(['-l' + l for l in v])}"
            for n, v in self.libraries.items()
        ]
        compiler = compiler or self.compiler
        string = f"{compiler} -std={self.std} "
        if self.link:
            string += " -shared "
        else:
            string += " -c "
        target = target or self.target
        string += (
            f"-o {target} {' '.join(self.sources)} "
            f"{' '.join(defines)} "
            f"{' '.join(includes)} "
            f"{self.cflags} {self.extra_cflags}"
            f"{' '.join(libraries)} "
            f"{self.lflags} {self.extra_lflags}"
        )
        return re.sub(r" +", r" ", string)


class Link:
    def __init__(self, outs, target, compiler="ld", build_directory: str = None):
        if not isinstance(outs, (list, tuple)):
            outs = [outs]
        if build_directory is not None:
            build_directory = Path(build_directory)
            new_outs = []
            for p in outs:
                if not Path(p).is_absolute():
                    new_outs.append(str(build_directory / p))
                else:
                    new_outs.append(p)
            outs = new_outs
            target = Path(target)
            if target.is_absolute():
                target = str(build_directory / target)
        self.outs = [str(p) for p in outs]
        self.target = str(target)
        self.compiler = compiler

    def shell(self, target: str = None):
        string = f"{self.compiler} -r "
        if target is None:
            target = self.target
        string += f"-o {target} {' '.join(self.outs)} "
        return string


class Nvcc(Gpp):
    def __init__(
        self,
        sources,
        target,
        arch=None,
        std="c++11",
        includes: list = None,
        defines: dict = None,
        cflags: str = None,
        extra_cflags: str = None,
        extra_lflags: str = None,
        build_directory: str = None,
    ):
        if arch is None:
            arch = find_cuda_device_arch()
            if arch is None:
                raise ValueError("you must specify arch if use cuda.")

        cflags = (
            cflags or f"-x cu -Xcompiler -fPIC -arch={arch} --expt-relaxed-constexpr"
        )
        try:
            cuda_home = find_cuda()
        except:
            cuda_home = None
        if cuda_home is not None:
            cuda_include = Path(cuda_home) / "include"
        includes = includes or []
        includes += [str(cuda_include)]
        super().__init__(
            sources,
            target,
            std,
            includes,
            defines,
            cflags,
            compiler="nvcc",
            extra_cflags=extra_cflags,
            extra_lflags=extra_lflags,
            build_directory=build_directory,
        )


class CUDALink(Gpp):
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
        includes = includes or []
        defines = defines or {}
        libraries = libraries or {}
        cflags = cflags or "-fPIC -O3"
        try:
            cuda_home = find_cuda()
        except:
            cuda_home = None
        if cuda_home is not None:
            cuda_include = Path(cuda_home) / "include"
            includes += [str(cuda_include)]
            cuda_lib_path = Path(cuda_home) / "lib64"
            cuda_libs = {str(cuda_lib_path): ["cublas", "cudart"]}
            libraries = {**libraries, **cuda_libs}
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


class NodeState(Enum):
    Evaled = "evaled"
    Normal = "normal"
    Error = "error"


class Node:
    def __init__(self, name=None):
        self.name = name
        self.prev = []
        self.next = []
        self.state = NodeState.Normal

    def __call__(self, *nodes):
        for node in nodes:
            self.prev.append(node)
            node.next.append(self)
        return self

    def _eval(self, *args, **kw):
        return True

    def eval(self, *args, **kw):
        for p in self.prev:
            if not p.eval(*args, **kw):
                self.state = NodeState.Error
                return False
        if self.state == NodeState.Normal:
            if self._eval(*args, **kw):
                self.state = NodeState.Evaled
            else:
                self.state = NodeState.Error
                return True
        return True

    def reset(self):
        self.state = NodeState.Normal
        self.prev = []
        self.next = []
        for node in self.prev:
            node.reset()


class TargetNode(Node):
    def __init__(self, srcs, hdrs, deps, copts, name=None):
        super().__init__(name)
        self.srcs = srcs
        self.hdrs = hdrs
        self.deps = deps
        self.copts = copts

    def _eval(self, executor):
        pass


def compile_func(cmd, code_folder, compiler):
    if not isinstance(cmd, (Link, Nvcc)):
        shell = cmd.shell(compiler=compiler)
    else:
        shell = cmd.shell()
    print(shell)
    cwd = None
    if code_folder is not None:
        cwd = str(code_folder)
    ret = subprocess.run(shell, shell=True, cwd=cwd)
    if ret.returncode != 0:
        raise RuntimeError("compile failed with retcode", ret.returncode)
    return ret


def compile_libraries(cmds, code_folder=None, compiler: str = None, num_workers=-1):
    if num_workers == -1:
        num_workers = min(len(cmds), multiprocessing.cpu_count())
    # for cmd in cmds:
    #     print(cmd.shell())
    if num_workers == 0:
        rets = map(
            partial(compile_func, code_folder=code_folder, compiler=compiler), cmds
        )
    else:
        with ProcessPoolExecutor(num_workers) as pool:
            func = partial(compile_func, code_folder=code_folder, compiler=compiler)
            rets = pool.map(func, cmds)

    if any([r.returncode != 0 for r in rets]):
        cmds.clear()
        return False
    cmds.clear()
    return True


def out(path):
    return Path(path).parent / (Path(path).stem + ".o")
