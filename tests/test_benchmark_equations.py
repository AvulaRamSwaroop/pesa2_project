
"""Independent regression checks for selected standard benchmark equations."""

import numpy as np
import pytest

from benchmarks import get_problem, implemented_names
from experiments.run import hypervolume_reference


@pytest.mark.parametrize(
    "name", ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6"]
)
def test_zdt_objectives_against_independent_equations(name):
    x = np.array([0.25, 0.10, 0.20, 0.30, 0.40])
    problem = get_problem(name, n_var=len(x))

    if name == "ZDT6":
        f1 = 1 - np.exp(-4 * x[0]) * np.sin(6 * np.pi * x[0])**6
        g = 1 + 9 * np.mean(x[1:])**0.25
        expected = [f1, g * (1 - (f1 / g)**2)]

    elif name == "ZDT4":
        g = (
            1
            + 10 * (len(x) - 1)
            + np.sum(
                x[1:]**2 - 10 * np.cos(4 * np.pi * x[1:])
            )
        )
        expected = [x[0], g * (1 - np.sqrt(x[0] / g))]

    else:
        g = 1 + 9 * np.mean(x[1:])

        if name == "ZDT1":
            h = 1 - np.sqrt(x[0] / g)
        elif name == "ZDT2":
            h = 1 - (x[0] / g)**2
        else:
            h = (
                1
                - np.sqrt(x[0] / g)
                - (x[0] / g) * np.sin(10 * np.pi * x[0])
            )

        expected = [x[0], g * h]

    assert np.allclose(
        problem.evaluate(x), expected, rtol=1e-12, atol=1e-12
    )


def test_dtlz1_objective_equation_and_front_hyperplane():
    problem = get_problem("DTLZ1", n_var=7, n_obj=3)
    x = np.array([0.2, 0.4, 0.6, 0.5, 0.5, 0.5, 0.5])

    g = 100 * (
        5
        + np.sum(
            (x[2:] - 0.5)**2
            - np.cos(20 * np.pi * (x[2:] - 0.5))
        )
    )

    expected = 0.5 * (1 + g) * np.array([
        x[0] * x[1],
        x[0] * (1 - x[1]),
        1 - x[0],
    ])

    assert np.allclose(
        problem.evaluate(x), expected, rtol=1e-12, atol=1e-12
    )
    assert np.allclose(problem.true_front(100).sum(axis=1), 0.5)


def test_fonseca_equation():
    problem = get_problem("FONSECA", n_var=3)
    x = np.array([0.1, -0.2, 0.3])
    n = len(x)

    expected = [
        1 - np.exp(-np.sum((x - 1 / np.sqrt(n))**2)),
        1 - np.exp(-np.sum((x + 1 / np.sqrt(n))**2)),
    ]

    assert np.allclose(
        problem.evaluate(x), expected, rtol=1e-12, atol=1e-12
    )


def test_kursawe_equation():
    problem = get_problem("KURSAWE", n_var=3)
    x = np.array([-1.0, 0.5, 2.0])

    f1 = sum(
        -10 * np.exp(
            -0.2 * np.sqrt(x[i]**2 + x[i + 1]**2)
        )
        for i in range(2)
    )
    f2 = np.sum(np.abs(x)**0.8 + 5 * np.sin(x**3))

    assert np.allclose(
        problem.evaluate(x), [f1, f2], rtol=1e-12, atol=1e-12
    )


def test_schaffer_equation_and_front():
    problem = get_problem("SCHAFFER")
    x = np.array([1.25])

    assert np.allclose(
        problem.evaluate(x), [x[0]**2, (x[0] - 2)**2]
    )

    front = problem.true_front(50)
    t = np.linspace(0, 2, 50)
    assert np.allclose(front, np.c_[t**2, (t - 2)**2])


def test_viennet_equation():
    problem = get_problem("VIENNET")
    a, b = 0.4, -0.7

    expected = [
        0.5 * (a*a + b*b) + np.sin(a*a + b*b),
        (3*a - 2*b + 4)**2 / 8 + (a - b + 1)**2 / 27 + 15,
        1 / (a*a + b*b + 1) - 1.1 * np.exp(-(a*a + b*b)),
    ]

    assert np.allclose(
        problem.evaluate(np.array([a, b])),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_hypervolume_reference_points_are_benchmark_specific():
    assert np.array_equal(
        hypervolume_reference("ZDT1", 2), [1.1, 1.1]
    )
    assert np.array_equal(
        hypervolume_reference("SCHAFFER", 2), [4.1, 4.1]
    )
    assert np.array_equal(
        hypervolume_reference("KURSAWE", 2), [1.0, 27.0]
    )
    assert hypervolume_reference("VIENNET", 3) is None


def test_all_registered_benchmarks_return_finite_values():
    for name in implemented_names():
        problem = (
            get_problem(name, n_obj=3)
            if name.startswith("DTLZ")
            else get_problem(name)
        )
        x = (problem.lower_bounds + problem.upper_bounds) / 2

        assert problem.evaluate(x).shape == (problem.n_obj,), name
        assert np.all(np.isfinite(problem.true_front(25))), name
