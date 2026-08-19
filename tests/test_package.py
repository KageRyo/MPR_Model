import wqsurrogatemodels


def test_public_namespace_exposes_direct_baseline():
    assert wqsurrogatemodels.__all__ == ["__version__", "categorize_score", "direct_wqi5_score"]
    assert wqsurrogatemodels.__version__ == "2.1.0"
    assert wqsurrogatemodels.direct_wqi5_score(7.2, 2.1, 0.3, 450, 12) == 59.921
