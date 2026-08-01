"""The three generated-text metrics, and the difference between them.

Hand-rolled metrics are worth distrusting, so these check known values rather than
just shapes: ROUGE-L and token-F1 disagree in a specific, predictable way (word
order), and the pair is reported together for exactly that reason. A test that
only asserted "returns a float between 0 and 1" would pass for a metric that
returned the same number every time.

No optional dependency here — that is part of the point. These are numpy-only, so
a seq2seq bundle can be scored in a serving image with no NLP stack in it.
"""

from __future__ import annotations

import math

import pytest

from ml_framework.core.metrics import exact_match, rouge_l, token_f1


# ── exact_match ───────────────────────────────────────────
@pytest.mark.unit
def test_exact_match_ignores_surrounding_whitespace():
    assert exact_match(["a summary"], ["  a summary  "]) == 1.0


@pytest.mark.unit
def test_exact_match_is_all_or_nothing_per_row():
    """One word wrong is a zero. That strictness is why it is reported last."""
    assert exact_match(["the cat sat"], ["the cat sags"]) == 0.0
    assert exact_match(["a", "b"], ["a", "wrong"]) == 0.5


# ── token_f1 vs rouge_l: the disagreement they exist for ──
@pytest.mark.unit
def test_token_f1_ignores_word_order_and_rouge_l_does_not():
    """The reason both are reported instead of either alone.

    A prediction with all the right words in the wrong order is perfect by
    multiset overlap and clearly not a correct sentence. ROUGE-L is the companion
    that notices.
    """
    reference, scrambled = ["the cat sat"], ["sat cat the"]

    assert token_f1(reference, scrambled) == 1.0
    assert rouge_l(reference, scrambled) < 1.0


@pytest.mark.unit
def test_rouge_l_rewards_a_subsequence_not_only_a_substring():
    """Right words, right order, extra words in between still scores well —
    which is what makes it usable for summarization rather than only for copying."""
    score = rouge_l(["the cat sat"], ["the big cat quietly sat"])
    assert 0.5 < score < 1.0


@pytest.mark.unit
def test_rouge_l_matches_a_hand_computed_value():
    """LCS = 3 of 4 reference words and 3 of 5 predicted.

    precision 3/5, recall 3/4, F = 2*0.6*0.75/(0.6+0.75) = 2/3.
    """
    score = rouge_l(["a b c d"], ["a x b c e"])
    assert math.isclose(score, 2 / 3, rel_tol=1e-9)


@pytest.mark.unit
def test_token_f1_matches_a_hand_computed_value():
    """Overlap 2 of 3 predicted and 2 of 4 reference → F = 2*(2/3)*(1/2)/(2/3+1/2)."""
    score = token_f1(["a b c d"], ["a b x"])
    assert math.isclose(score, 2 * (2 / 3) * 0.5 / (2 / 3 + 0.5), rel_tol=1e-9)


# ── the edges that would otherwise divide by zero ─────────
@pytest.mark.unit
def test_an_empty_prediction_scores_zero_rather_than_raising():
    """A model that learned to emit nothing is a real outcome, and it must score
    as one rather than crashing the evaluation that would have revealed it."""
    assert token_f1(["something"], [""]) == 0.0
    assert rouge_l(["something"], [""]) == 0.0


@pytest.mark.unit
def test_two_empty_strings_match():
    assert token_f1([""], [""]) == 1.0
    assert rouge_l([""], [""]) == 1.0


@pytest.mark.unit
def test_case_is_normalized_away():
    """A metric that changed with capitalization would rank models by their
    detokenizer rather than by their content."""
    assert rouge_l(["The Cat"], ["the cat"]) == 1.0


@pytest.mark.unit
def test_mismatched_counts_raise_rather_than_zipping_short():
    """Silently truncating to the shorter list would score a subset of the test
    set and report it as the whole thing."""
    with pytest.raises(ValueError, match="counts differ"):
        token_f1(["a", "b"], ["a"])


@pytest.mark.unit
def test_scores_average_over_rows():
    assert math.isclose(rouge_l(["a b", "c d"], ["a b", "x y"]), 0.5, rel_tol=1e-9)
