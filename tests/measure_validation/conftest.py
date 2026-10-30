import math
import shutil
from collections.abc import Generator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from quasar_models import GaussianModel, PowerLawModel
from quasar_models.modeling import PrepareModel
from quasar_typing.numpy import FloatMatrix, FloatVector, SortedFloatVector
from quasar_utils.setup import Info
from scipy.stats import gaussian_kde, norm

from quasar_errors.analysis import Analyser
from quasar_errors.error_result import ErrorResult
from quasar_errors.model_samples import (
    GaussianSampleList,
    PowerLawSample,
    Sample,
)

N: int = 1000

_this_file: Path = Path(__file__).resolve() 

FLUX: float = 100.0
ALPHA: float = -1.0

STRENGTH: float = 1000.0
FWHM_V: float = 5000.0
V_OFF: float = 0.0

###

DFLUX: float = 1.0
DALPHA: float = 0.01

DSTRENGTH: float = 1.0
DFWHM_V: float = 1.0
DV_OFF: float = 1.0

###

FLUX_BOUNDS: tuple[float, float] = (0, 1000)
ALPHA_BOUNDS: tuple[float, float] = (-2, 0)

STRENGTH_BOUNDS: tuple[float, float] = (0, 1e4)
FWHM_V_BOUNDS: tuple[float, float] = (1000, 10_000)
V_OFF_BOUNDS: tuple[float, float] = (-1000, 1000)

###

@pytest.fixture(scope="session")
def rest_wavelength() -> float:
    return 1549.0

    
@pytest.fixture(scope="session")
def info() -> Info:
    return Info()


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng()


@pytest.fixture(scope="session")
def powerlaw_models(rng: np.random.Generator) -> list[PowerLawModel]:
    fluxs = np.clip(rng.normal(FLUX, DFLUX, size=1+N), *FLUX_BOUNDS)
    alphas = np.clip(rng.normal(ALPHA, DALPHA, size=1+N), *ALPHA_BOUNDS)

    fluxs[0] = 100
    alphas[0] = -1

    return [
        PowerLawModel.create(
            1450, 1, *args,
            flux_bounds=FLUX_BOUNDS,
            alpha_bounds=ALPHA_BOUNDS,
        )
        for i, args in enumerate(zip(fluxs, alphas))
    ]


@pytest.fixture(scope="session")
def gaussian_models(rng: np.random.Generator, info: Info) -> list[GaussianModel]:
    strengths = np.clip(rng.normal(STRENGTH, DSTRENGTH, size=1+N), *STRENGTH_BOUNDS)
    fwhm_vs = np.clip(rng.normal(FWHM_V, DFWHM_V, size=1+N), *FWHM_V_BOUNDS)
    v_offs = np.clip(rng.normal(V_OFF, DV_OFF, size=1+N), *V_OFF_BOUNDS)

    strengths[0] = STRENGTH
    fwhm_vs[0] = FWHM_V
    v_offs[0] = V_OFF

    return [
        GaussianModel.create(
            1549.0, info.loading.sigma_res, "b",
            strength=s,
            fwhm_v=f,
            v_off=v,
            strength_bounds=STRENGTH_BOUNDS,
            fwhm_v_bounds=FWHM_V_BOUNDS,
            v_off_bounds=V_OFF_BOUNDS
        )
        for s, f, v in zip(strengths, fwhm_vs, v_offs)
    ]


@pytest.fixture(scope="session")
def x_array(info: Info) -> SortedFloatVector:
    x_bounds: tuple[float, float] = (1000.0, 1800.0)
    sigma_res = info.loading.sigma_res
    n = math.ceil(math.log(x_bounds[1] / x_bounds[0]) / math.log(1 + sigma_res))
    out = x_bounds[0] * (1 + sigma_res) ** np.arange(n + 1)
    assert out[-1] >= x_bounds[1]
    return out


@pytest.fixture(scope="session")
def dx_array(x_array: SortedFloatVector, info: Info) -> FloatVector:
    return x_array * info.loading.sigma_res


@pytest.fixture(scope="session")
def dy_array(x_array: SortedFloatVector) -> FloatVector:
    return np.ones_like(x_array)


@pytest.fixture(scope="session")
def noise_matrix(rng: np.random.Generator, dy_array: FloatVector) -> FloatMatrix:
    return np.stack(
        [rng.normal(0, dy_array) for _ in range(N + 1)],
        axis=0,
    )


@pytest.fixture(scope="session")
def y_matrix_pure_pl(
    x_array: SortedFloatVector,
    powerlaw_models: list[PowerLawModel],
) -> FloatMatrix:
    return np.stack([m(x_array) for m in powerlaw_models], axis=0)


@pytest.fixture(scope="session")
def y_matrix_pure_em(
    x_array: SortedFloatVector,
    gaussian_models: list[GaussianModel],
) -> FloatMatrix:
    return np.stack([m(x_array) for m in gaussian_models], axis=0)

@pytest.fixture(scope="session")
def y_matrix_pure(
    y_matrix_pure_pl: FloatMatrix,
    y_matrix_pure_em: FloatMatrix,
) -> FloatMatrix:
    return y_matrix_pure_pl + y_matrix_pure_em


@pytest.fixture(scope="session")
def y_matrix(
    y_matrix_pure: FloatMatrix,
    noise_matrix: FloatMatrix,
) -> FloatMatrix:
    return y_matrix_pure + noise_matrix


@pytest.fixture(scope="session")
def samples(
    x_array: SortedFloatVector,
    y_matrix: FloatMatrix,
    dy_array: FloatVector,
    dx_array: FloatVector,
    gaussian_models: list[GaussianModel],
    powerlaw_models: list[PowerLawModel],
) -> list[Sample]:
    out = []
    for y, pl_model, em_model in zip(y_matrix, powerlaw_models, gaussian_models):
        preparer = PrepareModel(x=x_array, model=pl_model + em_model)
        with preparer:
            out.append(
                Sample(
                    x_array, y, dy_array, dx_array,
                    pl=PowerLawSample.fromPowerLawModel(pl_model),
                    em=GaussianSampleList.fromGaussianSubmodels([em_model])
                )
            )
    return out


@pytest.fixture(scope="session")
def error_result(
    x_array: SortedFloatVector,
    samples: list[Sample],
    gaussian_models: list[GaussianModel],
    powerlaw_models: list[PowerLawModel],
) -> ErrorResult:
    pl_model = powerlaw_models[0]
    em_model = gaussian_models[0]
    preparer = PrepareModel(x=x_array, model=pl_model + em_model)
    with preparer as (seq_model, _):
        return ErrorResult(
            seq_model=seq_model, 
            samples=samples,
            sample_weights=tuple(1 for _ in samples),
        )


@pytest.fixture(scope="session")
def analyser(error_result: ErrorResult, info: Info) -> Analyser:
    return Analyser(
        error_result,
        info=info,
        fwhm_strategy="average",
        v_int=0,
        ipv_int=1,
    )

@pytest.fixture(scope="session")
def measures_dir() -> Generator[Path, None, None]:
    from quasar_io import load_measure_result

    p = _this_file.parent / "temp_measures"
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    yield p

    d = _this_file.parent / "temp_plots"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

    for path in p.glob("*_model"):
        mr_model = load_measure_result(p, path.name)
        mr_spectrum = load_measure_result(p, path.name.replace("_model", "_spectrum"))

        x = np.linspace(
            min(mr_model.val[1:].min(), mr_spectrum.val[1:].min()),
            max(mr_model.val[1:].max(), mr_spectrum.val[1:].max()),
            1000,
        )

        y_model = gaussian_kde(mr_model.val[1:])(x)
        y_spectrum = gaussian_kde(mr_spectrum.val[1:])(x)

        x_true = mr_model.val[0]
        x_first = mr_spectrum.val[0]
        dx_first = mr_spectrum.unc[0]

        measure = path.stem.split("_")[0]

        fig, ax = plt.subplots()
        ax.plot(
            x,
            y_model / y_model.max(),
            color="dodgerblue",
            label="Model (KDE)"
        )
        ax.axvline(x_true, color="lime", linestyle="--", label="True")
        
        ax.plot(
            x,
            y_spectrum / y_spectrum.max(),
            color="firebrick",
            label="Spectrum (KDE)"
        )
        if dx_first > 0.0:
            y_first = norm.pdf(x, loc=x_first, scale=dx_first)
            ax.plot(
                x,
                y_first / y_first.max(),
                color="firebrick",
                linestyle="dotted",
                label="Spectrum (Initial)"
            )
        else:
            ax.axvline(
                x_first, 
                color="firebrick", 
                linestyle="dotted", 
                label="Spectrum (Initial)",
            )
        ax.set_xlabel(measure)
        ax.set_ylabel("Peak-normalized PDF")
        ax.legend()
        fig.savefig(d / f"{measure}.png")
        plt.close()