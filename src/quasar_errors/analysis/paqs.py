import math
from typing import Any, Literal, Self

from astropy.units import Unit
from pydantic import BaseModel, field_validator, validate_call
from quasar_typing.re import StrPattern

from quasar_errors.measure_result import FrozenMeasureResult, MeasureResultDict

from .analyser import Analyser
from .conversions import WavelengthConversions


class PAQSContinuumOutput(MeasureResultDict):
    results: dict[
        Literal["l_1450", "l_2200", "l_3000", "l_5100"],
        FrozenMeasureResult,
    ]

    @field_validator("results")
    @classmethod
    def ensure_correct_units(
        cls,
        value: dict[str, FrozenMeasureResult | None],
    ) -> dict[str, FrozenMeasureResult | None]:
        return {
            k: v.to("erg/(s.angstrom)") 
            for k, v in value.items()
        }

    @field_validator("results")
    @classmethod
    def ensure_luminosities_are_log10(
        cls,
        value: dict[str, FrozenMeasureResult | None],
    ) -> dict[str, FrozenMeasureResult | None]:
        return {k: v.log(10) for k, v in value.items()}

    @classmethod
    def _fromAnalyser(
        cls, 
        analyser: Analyser, 
        from_model: bool,
    ) -> dict[Literal["results"], dict[str, FrozenMeasureResult]]:
        my_lb = analyser.x[0]
        my_ub = analyser.x[-1]

        distance_modulus = 4 * math.pi \
            * analyser.luminosity_distance_mpc ** 2 * Unit("Mpc^2")

        results = {}
        for key, w_bounds in {
            "l_1450": (1425.0, 1475.0),
            "l_2200": (2175.0, 2225.0),
            "l_3000": (2975.0, 3025.0),
            "l_5100": (5075.0, 5125.0),
        }.items():
            lb, ub = analyser.info.units.getWavelength(w_bounds * Unit("angstrom"))
            if (my_lb <= lb) and (ub <= my_ub):
                result = analyser.measureAverageFlux(
                    (lb, ub),
                    from_model=from_model,
                ) * distance_modulus
                results[key] = result
        
        return {"results": results}

    @classmethod
    @validate_call
    def fromAnalyser(
        cls, 
        analyser: Analyser, 
        *,
        from_model: bool = True,
    ) -> Self:
        return cls(**cls._fromAnalyser(analyser, from_model))


class PAQSLineOutput(MeasureResultDict):
    results: dict[
        Literal["CEN_REST", "CEN_OBS", "FLUX_OBS", "FWHM", "EW"], 
        FrozenMeasureResult,
    ]

    @field_validator("results")
    @classmethod
    def ensure_keys_present(
        cls,
        value: dict[str, FrozenMeasureResult],
    ) -> dict[str, FrozenMeasureResult]:
        for key in ("CEN_REST", "CEN_OBS", "FLUX_OBS", "FWHM", "EW"):
            if key not in value:
                raise ValueError(f"Missing key {key} in results")
        return value

    @field_validator("results")
    @classmethod
    def ensure_correct_units(
        cls,
        value: dict[str, FrozenMeasureResult],
    ) -> dict[str, FrozenMeasureResult]:
        for k, v in value.items():
            value[k] = v.to({
                "CEN_REST": "angstrom",
                "CEN_OBS": "angstrom",
                "FLUX_OBS": "erg/(s.cm^2)",
                "FWHM": "km/s",
                "EW": "angstrom",
            }[k])
        return value

    @field_validator("results")
    @classmethod
    def ensure_fluxs_are_log10(
        cls,
        value: dict[str, FrozenMeasureResult],
    ) -> dict[str, FrozenMeasureResult]:
        if "FLUX_OBS" in value:
            value["FLUX_OBS"] = value["FLUX_OBS"].log(10)
        return value

    @classmethod
    def _fromAnalyser(
        cls,
        analyser: Analyser,
        from_model: bool,
    ) -> dict[Literal["results"], dict[str, FrozenMeasureResult]]:
        results = {
            "CEN_REST": analyser.measureCentroid(from_model=from_model),
            "FLUX_OBS": analyser.measureStrength(from_model=from_model),
            "FWHM": analyser.measureFWHMVelocity(from_model=from_model),
            "EW": analyser.measureEW(from_model=from_model)
        }
        results["CEN_OBS"] = WavelengthConversions.rest_to_observed(
            results["CEN_REST"], 
            analyser.z,
        )
        return {"results": results}


    @classmethod
    @validate_call
    def fromAnalyser(
        cls, 
        analyser: Analyser, 
        *,
        from_model: bool = True,
    ) -> Self:
        return cls(**cls._fromAnalyser(analyser, from_model))


class PAQSOutput(BaseModel):
    continuum: PAQSContinuumOutput
    lya: PAQSLineOutput | None = None
    civ: PAQSLineOutput | None = None
    mgii: PAQSLineOutput | None = None
    hb: PAQSLineOutput | None = None
    o5007: PAQSLineOutput | None = None
    ha: PAQSLineOutput | None = None

    @classmethod
    def _fromAnalyser(
        cls,
        analyser: Analyser,
        from_model: bool,
        lya_pattern: StrPattern,
        civ_pattern: StrPattern,
        mgii_pattern: StrPattern,
        hb_pattern: StrPattern,
        o5007_pattern: StrPattern,
        ha_pattern: StrPattern,
    ) -> dict[str, dict[str, Any]]:
        out = {
            "continuum": PAQSContinuumOutput._fromAnalyser(analyser, from_model),
        }
        for key in ("lya", "civ", "mgii", "hb", "o5007", "ha"):
            pattern: StrPattern = locals()[f"{key}_pattern"]
            try:
                partial_analyser = Analyser.create(
                    analyser.error_result,
                    analyser.meta,
                    target_pattern=pattern,
                    pool=analyser.pool,
                    info=analyser.info,
                    res=analyser.res,
                    render_width=analyser.render_width,
                    fwhm_strategy=analyser.fwhm_strategy,
                    v_int=analyser.v_int,
                    ipv_int=analyser.ipv_int,
                    dx_int=analyser.dx_int,
                )
            except ValueError:
                continue

            out[key] = PAQSLineOutput._fromAnalyser(partial_analyser, from_model)

        return out

    @classmethod
    @validate_call
    def fromAnalyser(
        cls,
        analyser: Analyser,
        *,
        from_model: bool = True,
        lya_pattern: StrPattern = "bLyman-a.*",
        civ_pattern: StrPattern = "bCIV-1549.*",
        mgii_pattern: StrPattern = "bMgII-2798.*",
        hb_pattern: StrPattern = "bBalmer-b.*",
        o5007_pattern: StrPattern = "b[OIII]-5007.*",
        ha_pattern: StrPattern = "bBalmer-a.*",
    ) -> Self:
        return cls(**cls._fromAnalyser(
            analyser,
            from_model,
            lya_pattern,
            civ_pattern,
            mgii_pattern,
            hb_pattern,
            o5007_pattern,
            ha_pattern,
        ))

    def toFlattenedDict(self) -> dict[str, FrozenMeasureResult]:
        out = {
            k.upper(): v
            for k, v in self.continuum.results.items()
            if v is not None
        }
        for field_name in ("lya", "civ", "mgii", "hb", "o5007", "ha"):
            field_value = getattr(self, field_name, None)
            if field_value is None:
                continue
            out |= {
                f"{field_name}_{k}".upper(): v
                for k, v in field_value.results.items()
            }

        return out