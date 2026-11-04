import numpy as np
from benchmarks import get_problem
from algorithms.pesa2 import PESA2,PESA2Config
def test_zdt1_front():
 p=get_problem("ZDT1"); f=p.true_front(20)
 assert f.shape==(20,2)
 assert np.allclose(f[:,1],1-np.sqrt(f[:,0]))
def test_smoke():
 p=get_problem("ZDT1",10)
 r=PESA2(p,PESA2Config(20,20,3,10),seed=7).run()
 assert len(r["F"])>0 and r["evaluations"]==80
def test_dtlz2_shape():
 p=get_problem("DTLZ2",12,3)
 assert p.true_front(50).shape==(50,3)

def test_dtlz1_pareto_hyperplane():
 p=get_problem("DTLZ1",12,3)
 F=p.true_front(100)
 assert np.allclose(F.sum(axis=1),0.5)
def test_dtlz5_reference_manifold():
 p=get_problem("DTLZ5",12,3)
 F=p.true_front(100)
 assert np.allclose(F[:,0],F[:,1],atol=1e-6)

def test_hypervolume_2d_rectangle():
 from metrics.metrics import hv2
 # One point (1,1) and reference (3,3) covers area 4.
 assert abs(hv2(np.array([[1.,1.]]),np.array([3.,3.]))-4.0)<1e-12


def test_zdt4_standard_bounds():
 p=get_problem("ZDT4")
 assert p.lower_bounds[0] == 0 and p.upper_bounds[0] == 1
 assert np.all(p.lower_bounds[1:] == -5)
 assert np.all(p.upper_bounds[1:] == 5)

def test_hv_uses_supplied_reference_point():
 from metrics.metrics import compute_metrics
 F=np.array([[0.5,0.5]])
 R=np.array([[0.0,1.0],[1.0,0.0]])
 m=compute_metrics(F,R,0.1,10,hv_ref=np.array([2.0,2.0]))
 assert m["hv_reference_point"] == [2.0,2.0]


def test_zdt6_reference_front_is_pareto_objective_parameterized():
    p = get_problem("ZDT6")
    F = p.true_front(500)
    assert F.shape == (500, 2)
    assert np.isclose(F[0, 0], 0.2807753191, atol=1e-7)
    assert np.isclose(F[-1, 0], 1.0)
    assert np.allclose(F[:, 1], 1.0 - F[:, 0] ** 2)
    assert np.all(np.diff(F[:, 0]) > 0)

def test_all_implemented_benchmarks_return_finite_objectives_and_fronts():
    from benchmarks import implemented_names
    for name in implemented_names():
        p = get_problem(name, n_obj=3) if name.startswith("DTLZ") else get_problem(name)
        x = (p.lower_bounds + p.upper_bounds) / 2
        assert p.evaluate(x).shape == (p.n_obj,), name
        assert np.all(np.isfinite(p.true_front(25))), name

def test_standard_legacy_benchmarks_are_registered_and_finite():
    from benchmarks import implemented_names
    for name in ("FONSECA", "KURSAWE", "SCHAFFER", "VIENNET"):
        assert name in implemented_names()
        p = get_problem(name)
        x = (p.lower_bounds + p.upper_bounds) / 2
        assert np.all(np.isfinite(p.evaluate(x)))
        assert p.true_front(30).ndim == 2
        assert p.true_front(30).shape[1] == p.n_obj

def test_schaffer_analytical_front():
    p = get_problem("SCHAFFER")
    F = p.true_front(50)
    x = np.linspace(0.0, 2.0, 50)
    assert np.allclose(F[:, 0], x*x)
    assert np.allclose(F[:, 1], (x-2.0)**2)

def test_ambiguous_numbered_variants_are_not_guessed():
    import pytest
    for name in ("Binh (1)", "Binh (3)", "Deb (2)", "Fonseca (2)", "Rendon (2)", "Schaffer (2)", "Viennet (2)"):
        with pytest.raises(KeyError):
            get_problem(name)
