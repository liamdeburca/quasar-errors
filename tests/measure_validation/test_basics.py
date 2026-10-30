import numpy as np
import pytest
from quasar_typing.numpy import FloatMatrix, FloatVector

from quasar_errors.analysis import Analyser


def test_background_subtraction(
    subtests: pytest.Subtests,
    analyser: Analyser,
    y_matrix_pure_pl: FloatMatrix,
    y_matrix_pure_em: FloatMatrix,
    noise_matrix: FloatMatrix
):
    y_wo_pl = analyser.getFlux(
        bg_flux={"pl"},
        ensure_not_negative=False,
        simplify=False,
    )
    for i, (y1, y2, n) in enumerate(zip(y_wo_pl, y_matrix_pure_em, noise_matrix)):
        with subtests.test(i=i):
            np.testing.assert_allclose(y1, y2 + n)

    y_wo_em = analyser.getFlux(
        bg_flux={"em"},
        ensure_not_negative=False,
        simplify=False,
    )
    for i, (y1, y2, n) in enumerate(zip(y_wo_em, y_matrix_pure_pl, noise_matrix)):
        with subtests.test(i=i):
            np.testing.assert_allclose(y1, y2 + n)


def test_moment_params(
    analyser: Analyser,
    x_array: FloatVector,
    dx_array: FloatVector,
) -> None:
    mp = analyser.getMomentParams(
        from_model=True,
        bg_flux={"pl"},
        centre="wave",
    )
    np.testing.assert_array_equal(mp.xs, x_array[None,:])
    np.testing.assert_array_equal(mp.dxs, dx_array[None,:])
