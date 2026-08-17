import wqsurrogatemodels


def test_public_namespace_exposes_direct_baseline():
    assert wqsurrogatemodels.__all__ == ["categorize_score", "direct_wqi5_score"]
    assert wqsurrogatemodels.direct_wqi5_score(7.2, 2.1, 0.3, 450, 12) == 59.921
