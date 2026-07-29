__all__ = [
    "fourth_moment",
    "second_moment",
    "third_moment",
    "weighted_fourth_moment",
    "weighted_second_moment",
    "weighted_third_moment",
]

from numpy import dot
from quasar_typing.numpy import FloatVector


def second_moment(
    first_moment: float | FloatVector,
    mu: float | FloatVector,
    sigma: float | FloatVector,
) -> float | FloatVector:
    return sigma**2 + (mu - first_moment) ** 2


def weighted_second_moment(
    weights: FloatVector,
    first_moment: float | FloatVector,
    mu: float | FloatVector,
    sigma: float | FloatVector,
) -> float | FloatVector:
    return dot(weights, sigma**2 + (mu - first_moment) ** 2)


def third_moment(
    first_moment: float | FloatVector,
    second_moment: float | FloatVector,
    mu: float | FloatVector,
    sigma: float | FloatVector,
) -> float | FloatVector:
    return (
        3 * mu * sigma**2
        + mu**3
        - 2 * first_moment * (mu**2 + sigma**2)
        + mu * first_moment**2
        - first_moment * second_moment
    )


def weighted_third_moment(
    weights: FloatVector,
    first_moment: float | FloatVector,
    second_moment: float | FloatVector,
    mu: float | FloatVector,
    sigma: float | FloatVector,
) -> float | FloatVector:
    return dot(
        weights,
        third_moment(first_moment, second_moment, mu, sigma),
    )


def fourth_moment(
    first_moment: float | FloatVector,
    third_moment: float | FloatVector,
    mu: float | FloatVector,
    sigma: float | FloatVector,
) -> float | FloatVector:
    mu_sq = mu**2
    sigma_sq = sigma**2

    result = mu_sq**2 + 6 * mu_sq * sigma_sq + 3 * sigma_sq**2
    result -= 3 * first_moment * mu * (mu_sq + 3 * sigma_sq)
    result += 3 * first_moment**2 * (mu_sq + sigma_sq)
    result -= first_moment**3 * mu
    result -= first_moment * third_moment

    return result


def weighted_fourth_moment(
    weights: FloatVector,
    first_moment: float | FloatVector,
    third_moment: float | FloatVector,
    mu: float | FloatVector,
    sigma: float | FloatVector,
) -> float | FloatVector:
    return dot(
        weights,
        fourth_moment(first_moment, third_moment, mu, sigma),
    )
