"""engine/intelligence/tests/test_classifier.py -- Track B, Tasks 7-8 (FR-05.1, FR-05.2)."""

from __future__ import annotations

from engine.intelligence.router.classifier import RoutingDecision, SemanticRouterClassifier
from engine.intelligence.router.coldstart_data import RoutingExample
from engine.intelligence.tests.fakes import FakeEmbedder


def _fitted_router() -> SemanticRouterClassifier:
    router = SemanticRouterClassifier(FakeEmbedder())
    router.fit()  # trains on BOOTSTRAP_EXAMPLES by default
    return router


def test_fit_with_no_examples_uses_bootstrap_dataset() -> None:
    router = _fitted_router()
    assert router.is_fitted


def test_fit_raises_on_empty_example_list() -> None:
    router = SemanticRouterClassifier(FakeEmbedder())
    try:
        router.fit(examples=[])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_classify_returns_one_of_the_three_labels() -> None:
    router = _fitted_router()
    decision = router.classify("hi there, how are you")
    assert isinstance(decision, RoutingDecision)
    assert decision.tier in ("local", "cloud_byok", "cascade_cloud")


def test_classify_confidence_is_calibrated_probability() -> None:
    router = _fitted_router()
    decision = router.classify("hello")
    assert 0.0 <= decision.confidence <= 1.0


def test_classify_auto_fits_if_never_fitted_explicitly() -> None:
    router = SemanticRouterClassifier(FakeEmbedder())
    decision = router.classify("hello there")
    assert router.is_fitted
    assert decision.tier in ("local", "cloud_byok", "cascade_cloud")


def test_classify_trivial_query_routes_local_on_clean_training_set() -> None:
    """
    A minimal, unambiguous training set (distinct vocabularies per
    class) should let even a FakeEmbedder-backed classifier separate
    an exact-match trivial query into "local" -- this tests the
    classifier's *mechanics* (fit -> predict_proba -> argmax), not
    real-world routing accuracy, which is a Section 12 benchmark
    concern once real cold-start data exists.
    """
    examples = [
        RoutingExample("hello there friend", "local"),
        RoutingExample("hi how are you", "local"),
        RoutingExample("good morning to you", "local"),
        RoutingExample("write a compiler in rust", "cloud_byok"),
        RoutingExample("prove this theorem rigorously", "cloud_byok"),
        RoutingExample("optimize this algorithm please", "cloud_byok"),
        RoutingExample("what is today's date now", "cascade_cloud"),
        RoutingExample("search the web for current news", "cascade_cloud"),
        RoutingExample("check my current account balance", "cascade_cloud"),
    ]
    router = SemanticRouterClassifier(FakeEmbedder())
    router.fit(examples)

    decision = router.classify("hello there friend")
    assert decision.tier == "local"


def test_classify_is_deterministic_given_same_training_and_query() -> None:
    router_a = _fitted_router()
    router_b = _fitted_router()
    assert router_a.classify("hello").tier == router_b.classify("hello").tier
