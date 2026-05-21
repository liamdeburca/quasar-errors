from setuptools import setup, find_packages
from Cython.Build import cythonize
from numpy import get_include

setup(
    name="quasar_errors",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize(
        ["src/quasar_errors/measures/approx/approx.pyx"],
        language_level="3",
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
    include_dirs=[get_include()],
)
