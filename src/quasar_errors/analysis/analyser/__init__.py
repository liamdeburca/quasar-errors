from pydantic.dataclasses import dataclass

from .discrete_analyser import DiscreteAnalyser
from .distribution_analyser import DistributionAnalyser
from .gof_analyser import GoFAnalyser
from .integral_analyser import IntegralAnalyser
from .moments_analyser import MomentsAnalyser
from .quantiles_analyser import QuantilesAnalyser


@dataclass
class Analyser(
    DiscreteAnalyser,
    DistributionAnalyser,
    IntegralAnalyser,
    MomentsAnalyser, 
    QuantilesAnalyser,
    GoFAnalyser,
): 
    """
    Analyser class that combines the functionalities of:
    - `DiscreteAnalyser`
    - `DistributionAnalyser`
    - `IntegralAnalyser`
    - `MomentsAnalyser`
    - `QuantilesAnalyser`
    - `GoFAnalyser`
    """
    def __hash__(self) -> int:
        return hash(self.__tuple__())