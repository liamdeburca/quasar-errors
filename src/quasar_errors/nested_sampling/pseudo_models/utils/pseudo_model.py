from typing import ClassVar, Callable
from functools import cached_property
from numpy import array, float64
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass

from .pseudo_param import PseudoParam

@dataclass
class PseudoModel:
    param_names: ClassVar[list[str]]
    n_params: ClassVar[int]

    def __getstate__(self) -> dict:
        return self._parameters
    
    def __setstate__(self, state: dict) -> None:
        for param_name, param in state.items():
            setattr(self, param_name, param)

    @cached_property
    def _parameters(self) -> dict[str, PseudoParam]:
        """
        Returns a dictionary of all PseudoParam instances in the model.

        As changes to each PseudoParam are made inplace, this property can be
        safely cached.
        """
        return {
            param_name: getattr(self, param_name) 
            for param_name in self.param_names
        }

    @property
    def parameters(self) -> NDArray[float64]:
        """
        Returns an array of all parameter values in the model.
        """
        return array(
            [param.value for param in self._parameters.values()],
            dtype=float64,
        )
    
    @property
    def fixed(self) -> dict[str, bool]:
        """
        Returns a dictionary indicating which parameters are fixed.
        """
        return {
            param_name: param.fixed 
            for param_name, param in self._parameters.items()
        }
    
    @property
    def tied(self) -> dict[str, Callable | None]:
        """
        Returns a dictionary indicating which parameters are tied.
        """
        return {
            param_name: param.tied 
            for param_name, param in self._parameters.items()
        }
    
    @property
    def free(self) -> dict[str, bool]:
        """
        Returns a dictionary indicating which parameters are free.
        """
        return {
            param_name: param.free 
            for param_name, param in self._parameters.items()
        }

    def setParameters(self, parameters: NDArray[float64]) -> None:
        """
        Sets the values of all parameters in the model.
        """
        assert parameters.size == self.n_params
        for param, value in zip(self._parameters.values(), parameters):
            param.value = value
