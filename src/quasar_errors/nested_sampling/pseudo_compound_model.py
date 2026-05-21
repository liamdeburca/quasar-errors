from typing import Self, Callable, Literal
from dataclasses import field
from numpy import array, float64, zeros, bool_, stack, zeros_like
from functools import cached_property
from pydantic.dataclasses import dataclass

from quasar_typing.numpy import (
    FloatVector, BoolVector, FloatMatrix, FittableCoordsTuple,
)
from quasar_typing.astropy import Model_

from quasar_models import PowerLawModel, IronModel, BalmerModel, GaussianModel
from quasar_models.line.vprofilecopies import (
    VProfileCopy1G,
    VProfileCopy2G,
    VProfileCopy3G,
    VProfileCopy4G,
    VProfileCopy5G,
)
from .pseudo_models.utils import PseudoModel
from .pseudo_models import (
    PseudoPowerLawModel,
    PseudoIronModel,
    PseudoBalmerModel,
    PseudoGaussianModel,
    PseudoVProfileCopy1G,
    PseudoVProfileCopy2G,
    PseudoVProfileCopy3G,
    PseudoVProfileCopy4G,
    PseudoVProfileCopy5G,
)
from . import loss_functions
from .pseudo_models.utils.pseudo_param import PseudoParam

@dataclass
class PseudoCompoundModel:
    submodels: list[PseudoModel]
    
    data: FittableCoordsTuple | None = field(default=None, repr=False, kw_only=True)
    loss: Literal['chi2'] = field(default='chi2', kw_only=True)
    
    names: set[str] = field(init=False, default_factory=set, repr=False)
    name_to_submodel_dict: dict[str, PseudoModel] = field(init=False, repr=False)
    normalisation: float = field(init=False, repr=False)

    def __post_init__(self):
        self.names = set()
        self.name_to_submodel_dict = {}
        for submodel in self.submodels:
            name = submodel.name
            if name in self.names:
                msg = f"Duplicate submodel name: {name}"
                raise ValueError(msg)
            
            self.names.add(name)
            self.name_to_submodel_dict[name] = submodel

        if self.data is not None:
            self.setLoss(self.loss)

    def __getstate__(self):
        return {
            'submodels': self.submodels,
            'data': self.data,
            'loss': self.loss,
        }
    
    def __setstate__(self, state: dict) -> None:
        self.__init__(
            state['submodels'],
            data=state['data'],
            loss=state['loss'],
        )
        self.__post_init__()
    
    @classmethod
    def from_model(cls, model: Model_) -> Self:
        if model.n_submodels == 1:
            submodels = (model,)
        else:
            submodels = model

        pseudo_models = []
        for submodel in submodels:
            if isinstance(submodel, PowerLawModel):
                pseudo_model_class = PseudoPowerLawModel
            elif isinstance(submodel, IronModel):
                pseudo_model_class = PseudoIronModel
            elif isinstance(submodel, BalmerModel):
                pseudo_model_class = PseudoBalmerModel
            elif isinstance(submodel, GaussianModel):
                pseudo_model_class = PseudoGaussianModel
            elif isinstance(submodel, VProfileCopy1G):
                pseudo_model_class = PseudoVProfileCopy1G
            elif isinstance(submodel, VProfileCopy2G):
                pseudo_model_class = PseudoVProfileCopy2G
            elif isinstance(submodel, VProfileCopy3G):
                pseudo_model_class = PseudoVProfileCopy3G
            elif isinstance(submodel, VProfileCopy4G):
                pseudo_model_class = PseudoVProfileCopy4G
            elif isinstance(submodel, VProfileCopy5G):
                pseudo_model_class = PseudoVProfileCopy5G
            else:
                msg = f"Unsupported submodel type: {type(submodel).__name__}"
                raise ValueError(msg)

            pseudo_models.append(pseudo_model_class.from_model(submodel))

        return PseudoCompoundModel(pseudo_models)
    
    def __getitem__(self, sel: int | slice | str) -> PseudoModel | Self:
        if isinstance(sel, int):
            return self.submodels[sel]
        elif isinstance(sel, slice):
            return PseudoCompoundModel(self.submodels[sel])
        else:
            return self.name_to_submodel_dict[sel]
        
    @cached_property
    def _parameters(self) -> dict[str, PseudoParam]:
        _parameters = {}
        for i, submodel in enumerate(self.submodels):
            _parameters.update({
                f"{param_name}_{i}": param 
                for param_name, param in submodel._parameters.items()
            })
        return _parameters
        
    @cached_property
    def n_params(self) -> int:
        return len(self._parameters)

    @cached_property
    def param_names(self) -> list[str]:
        return list(self._parameters.keys())
    
    @cached_property
    def parameters(self) -> FloatVector:
        return array(
            [param.value for param in self._parameters.values()],
            dtype=float64,
        )
    
    @cached_property
    def fixed_indices(self) -> set[int]:
        return set(
            i
            for i, param in enumerate(self._parameters.values())
            if param.fixed
        )
    
    @cached_property
    def fixed_mask(self) -> BoolVector:
        mask = zeros(self.n_params, dtype=bool_)
        for idx in self.fixed_indices:
            mask[idx] = True
        return mask
    
    @cached_property
    def n_fixed(self) -> int: 
        return len(self.fixed_indices)

    @cached_property
    def tied_functions(self) -> dict[int, Callable[[Self], float]]:
        tied_functions = {}
        for i, param in enumerate(self._parameters.values()):
            if param.tied is not None:
                tied_functions[i] = param.tied
        return tied_functions    
    
    @cached_property
    def tied_indices(self) -> set[int]:
        return set(self.tied_functions.keys())
    
    @cached_property
    def tied_mask(self) -> BoolVector:
        mask = zeros(self.n_params, dtype=bool_)
        for idx in self.tied_indices:
            mask[idx] = True
        return mask
    
    @cached_property
    def n_tied(self) -> int: 
        return len(self.tied_indices)
    
    @cached_property
    def free_parameters(self) -> dict[int, PseudoParam]:
        return {
            i: param
            for i, param in enumerate(self._parameters.values())
            if param.free
        }
    
    @cached_property
    def free_indices(self) -> set[int]:
        return set(self.free_parameters.keys())
    
    @cached_property
    def free_mask(self) -> BoolVector:
        mask = zeros(self.n_params, dtype=bool_)
        for idx in self.free_indices:
            mask[idx] = True
        return mask
    
    @cached_property
    def n_free(self) -> int: 
        return len(self.free_indices)
        
    def setParameters(
        self,
        parameters: FloatVector,
    ) -> None:
        assert parameters.size == self.n_params
        del self.parameters # Clear cached property
        count: int = 0
        for submodel in self.submodels:
            n_params = submodel.n_params
            submodel.setParameters(parameters[count:count+n_params])
            count += n_params

    def upsample(self, p: FloatVector | FloatMatrix) -> FloatVector:
        assert p.shape[-1] == self.n_free
        
        if p.ndim == 1:
            p_upsampled = self.parameters.copy()
            p_upsampled[self.free_mask] = p

            if self.n_tied > 0:
                self.setParameters(p_upsampled)
                for idx, func in self.tied_functions.items():
                    p_upsampled[idx] = func(self)
                self.setParameters(p_upsampled)
            
            return p_upsampled
        
        return stack(list(map(self.upsample, p)), axis=0)
    
    def forward(
        self,
        x: FloatVector,
        parameters: FloatVector,
    ) -> FloatVector:
        assert parameters.size == self.n_params

        y = zeros_like(x, dtype=float64)
        count: int = 0
        for submodel in self.submodels:
            submodel(x, *parameters[count:count+submodel.n_params], y=y)
            count += submodel.n_params
        
        return y
        
    def setData(self, data: FittableCoordsTuple) -> None:
        self.data = data

    def setLoss(self, loss: Literal['chi2'] = 'chi2') -> None:
        normalisation_func = getattr(loss_functions, loss).normalisation
        self.loss = loss
        self.normalisation = normalisation_func(self.data[2])

    ### Uniform sampling

    @cached_property
    def free_lbs(self) -> FloatVector:
        return array(
            [param.bounds[0] for param in self.free_parameters.values()],
            dtype=float64,
        )
    
    @cached_property
    def free_ubs(self) -> FloatVector:
        return array(
            [param.bounds[1] for param in self.free_parameters.values()],
            dtype=float64,
        )