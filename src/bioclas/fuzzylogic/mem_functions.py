"""A module containing common membership functions for fuzzy sets."""

import numpy as np


def trimf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Triangular membership function.

    Args:
        x (np.ndarray): Input values.
        a (float): Left vertex of the triangle.
        b (float): Peak of the triangle.
        c (float): Right vertex of the triangle.

    Returns:
        np.ndarray: Membership values.
    """
    assert (
        a <= b <= c
    ), "Invalid parameters for triangular membership function. Must satisfy a <= b <= c."
    assert isinstance(x, np.ndarray), "Input must be a numpy array."
    assert x.ndim == 1, "Input array must be one-dimensional."
    assert np.issubdtype(
        x.dtype, np.number
    ), "Input array must contain numeric values."

    with np.errstate(divide="ignore", invalid="ignore"):
        left = np.where((b - a) == 0, 1, (x - a) / (b - a))
        right = np.where((c - b) == 0, 1, (c - x) / (c - b))
    return np.clip(np.minimum(left, right), 0, 1)


def trapmf(
    x: np.ndarray, a: float, b: float, c: float, d: float
) -> np.ndarray:
    """Trapezoidal membership function.

    Args:
        x (np.ndarray): Input values.
        a (float): Left foot of the trapezoid.
        b (float): Left shoulder of the trapezoid.
        c (float): Right shoulder of the trapezoid.
        d (float): Right foot of the trapezoid.

    Returns:
        np.ndarray: Membership values.
    """
    if not (a <= b <= c <= d):
        raise ValueError(
            f"Invalid parameters for trapezoidal membership function. Must satisfy a <= b <= c <= d. Given a={a}, b={b}, c={c}, d={d}"
        )
    assert (
        a <= b <= c <= d
    ), "Invalid parameters for trapezoidal membership function. Must satisfy a <= b <= c <= d."
    assert isinstance(x, np.ndarray), "Input must be a numpy array."
    assert x.ndim == 1, "Input array must be one-dimensional."
    assert np.issubdtype(
        x.dtype, np.number
    ), "Input array must contain numeric values."

    # divisions by 0 should output 1 membership where appropriate
    with np.errstate(divide="ignore", invalid="ignore"):
        left = np.where((b - a) == 0, 1, (x - a) / (b - a))
        right = np.where((d - c) == 0, 1, (d - x) / (d - c))
    return np.clip(np.minimum(np.minimum(left, 1), right), 0, 1)


def sigmf(x: np.ndarray, a: float, c: float) -> np.ndarray:
    """Sigmoidal membership function.

    Args:
        x (np.ndarray): Input values.
        a (float): Slope of the sigmoid.
        c (float): Center of the sigmoid.

    Returns:
        np.ndarray: Membership values.
    """
    assert isinstance(x, np.ndarray), "Input must be a numpy array."
    assert x.ndim == 1, "Input array must be one-dimensional."
    assert np.issubdtype(
        x.dtype, np.number
    ), "Input array must contain numeric values."
    return 1 / (1 + np.exp(-a * (x - c)))


def smf(x: np.ndarray, a: float, b) -> np.ndarray:
    """S-shaped membership function.

    Args:
        x (np.ndarray): Input values.
        a (float): Start of the S-shape.
        b (float): End of the S-shape.

    Returns:
        np.ndarray: Membership values.
    """
    assert (
        a < b
    ), "Invalid parameters for S-shaped membership function. Must satisfy a < b."
    assert isinstance(x, np.ndarray), "Input must be a numpy array."
    assert x.ndim == 1, "Input array must be one-dimensional."
    assert np.issubdtype(
        x.dtype, np.number
    ), "Input array must contain numeric values."

    y = np.zeros_like(x)
    idx1 = x <= a
    idx2 = (x > a) & (x < (a + b) / 2)
    idx3 = (x >= (a + b) / 2) & (x < b)
    idx4 = x >= b

    y[idx1] = 0
    y[idx2] = 2 * ((x[idx2] - a) / (b - a)) ** 2
    y[idx3] = 1 - 2 * ((b - x[idx3]) / (b - a)) ** 2
    y[idx4] = 1

    return y


def pimf(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Pi-shaped membership function.

    Args:
        x (np.ndarray): Input values.
        a (float): Left foot of the Pi-shape.
        b (float): Left shoulder of the Pi-shape.
        c (float): Right shoulder of the Pi-shape.
        d (float): Right foot of the Pi-shape.

    Returns:
        np.ndarray: Membership values.
    """
    assert (
        a <= b <= c <= d
    ), "Invalid parameters for Pi-shaped membership function. Must satisfy a <= b <= c <= d."
    assert isinstance(x, np.ndarray), "Input must be a numpy array."
    assert x.ndim == 1, "Input array must be one-dimensional."
    assert np.issubdtype(
        x.dtype, np.number
    ), "Input array must contain numeric values."

    y = np.zeros_like(x)
    idx1 = x <= a
    idx2 = (x > a) & (x < (a + b) / 2)
    idx3 = (x >= (a + b) / 2) & (x < b)
    idx4 = (x >= b) & (x <= c)
    idx5 = (x > c) & (x < (c + d) / 2)
    idx6 = (x >= (c + d) / 2) & (x < d)
    idx7 = x >= d

    y[idx1] = 0
    y[idx2] = 2 * ((x[idx2] - a) / (b - a)) ** 2
    y[idx3] = 1 - 2 * ((b - x[idx3]) / (b - a)) ** 2
    y[idx4] = 1
    y[idx5] = 1 - 2 * ((x[idx5] - c) / (d - c)) ** 2
    y[idx6] = 2 * ((d - x[idx6]) / (d - c)) ** 2
    y[idx7] = 0

    return y


if __name__ == "__main__":
    x = np.linspace(0, 10, 100)
    y = trimf(x, 2, 5, 8)
    print(y)  # Example output

    y2 = trapmf(x, 3, 3, 7, 7)
    print(y2)  # Example output
