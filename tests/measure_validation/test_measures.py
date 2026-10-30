from pathlib import Path

import pytest

from quasar_errors.analysis import Analyser, FrozenMeasureResult


def dump_result(
    measures_dir: Path,
    res: FrozenMeasureResult,
    measure: str,
    from_model: bool,
) -> None:
    from quasar_io import dump_measure_result
    dump_measure_result(
        measures_dir, res, 
        name=f"{measure}_{'model' if from_model else 'spectrum'}",
        json_ext=".toon",
        exist_ok=True,
    )

###


@pytest.mark.parametrize("from_model", [True, False])
def test_strength(from_model: bool, analyser: Analyser, measures_dir: Path) -> None:
    res = analyser.measureStrength(from_model=from_model)
    dump_result(measures_dir, res, "strength", from_model)


@pytest.mark.parametrize("from_model", [True, False])
def test_centroid(from_model: bool, analyser: Analyser, measures_dir: Path) -> None:
    res = analyser.measureCentroid(from_model=from_model)
    dump_result(measures_dir, res, "centroid", from_model)


@pytest.mark.parametrize("from_model", [True, False])
def test_dispersion(from_model: bool, analyser: Analyser, measures_dir: Path) -> None:
    res = analyser.measureDispersion(from_model=from_model)
    dump_result(measures_dir, res, "dispersion", from_model)


@pytest.mark.parametrize("from_model", [True, False])
def test_skewness(from_model: bool, analyser: Analyser, measures_dir: Path) -> None:
    res = analyser.measureSkewness(from_model=from_model)
    dump_result(measures_dir, res, "skewness", from_model)


@pytest.mark.parametrize("from_model", [True, False])
def test_kurtosis(from_model: bool, analyser: Analyser, measures_dir: Path) -> None:
    res = analyser.measureKurtosis(from_model=from_model)
    dump_result(measures_dir, res, "kurtosis", from_model)


@pytest.mark.parametrize("from_model", [True, False])
def test_median(from_model: bool, analyser: Analyser, measures_dir: Path) -> None:
    res = analyser.measureMedian(from_model=from_model)
    dump_result(measures_dir, res, "median", from_model)


@pytest.mark.parametrize("from_model", [True, False])
def test_fwhm_velocity(from_model: bool, analyser: Analyser, measures_dir: Path) -> None:
    res = analyser.measureFWHMVelocity(from_model=from_model)
    dump_result(measures_dir, res, "fwhm", from_model)


@pytest.mark.parametrize("from_model", [True, False])
def test_pl_lum(from_model: bool, analyser: Analyser, measures_dir: Path) -> None:
    res = analyser.measureIntegratedFlux(
        (1000, 1400),
        from_model=from_model,
        bg_flux={"em"},
    )
    dump_result(measures_dir, res, "plLum", from_model)


@pytest.mark.parametrize("from_model", [True, False])
def test_em_lum(from_model: bool, analyser: Analyser, measures_dir: Path) -> None:
    res = analyser.measureIntegratedFlux(
        (1400, 1800),
        from_model=from_model,
        bg_flux={"em"},
    )
    dump_result(measures_dir, res, "emLum", from_model)


@pytest.mark.parametrize("from_model", [True, False])
def test_ew(from_model: bool, analyser: Analyser, measures_dir: Path) -> None:
    res = analyser.measureEW(from_model=from_model)
    dump_result(measures_dir, res, "ew", from_model)
