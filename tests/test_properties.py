"""Property-based tests for the ier library using Hypothesis.

These tests verify mathematical properties and invariants that should hold
for any valid input, providing more comprehensive coverage than example-based tests.
"""

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from ier.acquiescence import acquiescence
from ier.evenodd import evenodd
from ier.irv import irv
from ier.longstring import (
    avgstr_message,
    longstr_message,
    longstring,
    run_length_decode,
    run_length_encode,
)
from ier.mahad import mahad
from ier.psychsyn import psychsyn
from ier.screen import screen


def valid_survey_data(
    min_rows: int = 3, max_rows: int = 50, min_cols: int = 2, max_cols: int = 20
) -> st.SearchStrategy[np.ndarray]:
    """Generate valid survey response data matrices."""
    return arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=min_rows, max_value=max_rows),
            st.integers(min_value=min_cols, max_value=max_cols),
        ),
        elements=st.floats(min_value=1, max_value=7, allow_nan=False, allow_infinity=False),
    )


def non_empty_string(min_size: int = 1, max_size: int = 100) -> st.SearchStrategy[str]:
    """Generate non-empty strings of printable characters."""
    return st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        min_size=min_size,
        max_size=max_size,
    )


class TestLongstringProperties:
    """Property-based tests for longstring module."""

    @given(st.text(min_size=0, max_size=100))
    def test_run_length_encode_decode_roundtrip(self, message: str) -> None:
        """Encoding then decoding should return the original string."""
        encoded = run_length_encode(message)
        decoded = run_length_decode(encoded)
        assert decoded == message

    @given(non_empty_string())
    def test_longstr_length_bounds(self, message: str) -> None:
        """Longest string length should be between 1 and total length."""
        result = longstr_message(message)
        assert result is not None
        char, length = result
        assert 1 <= length <= len(message)
        assert char in message

    @given(non_empty_string())
    def test_avgstr_bounds(self, message: str) -> None:
        """Average string length should be between 1 and total length."""
        result = avgstr_message(message)
        assert 1.0 <= result <= len(message)

    @given(non_empty_string())
    def test_longstr_greater_or_equal_avgstr(self, message: str) -> None:
        """Longest string length should be >= average string length."""
        longstr_result = longstr_message(message)
        avgstr_result = avgstr_message(message)
        assert longstr_result is not None
        assert longstr_result[1] >= avgstr_result

    @given(st.lists(non_empty_string(), min_size=1, max_size=20))
    def test_longstring_list_length_preserved(self, messages: list[str]) -> None:
        """Longstring on list should return same number of results."""
        result = longstring(messages)
        assert isinstance(result, list)
        assert len(result) == len(messages)

    @given(st.integers(min_value=1, max_value=100), st.sampled_from("ABCDEFGHIJ"))
    def test_repeated_char_longstring(self, repeat: int, char: str) -> None:
        """A repeated character string should have longest run equal to string length."""
        message = char * repeat
        result = longstr_message(message)
        assert result == (char, repeat)
        assert avgstr_message(message) == float(repeat)


class TestIRVProperties:
    """Property-based tests for IRV function."""

    @given(valid_survey_data())
    @settings(max_examples=50)
    def test_irv_non_negative(self, data: np.ndarray) -> None:
        """IRV (standard deviation) should always be non-negative."""
        result = irv(data)
        assert len(result) == data.shape[0]
        assert np.all(result >= 0)

    @given(valid_survey_data())
    @settings(max_examples=50)
    def test_irv_constant_rows_zero(self, data: np.ndarray) -> None:
        """Rows with constant values should have IRV of 0."""
        constant_data = np.ones_like(data) * 5
        result = irv(constant_data)
        np.testing.assert_array_almost_equal(result, np.zeros(data.shape[0]))

    @given(valid_survey_data(min_cols=4, max_cols=20))
    @settings(max_examples=30)
    def test_irv_split_same_length(self, data: np.ndarray) -> None:
        """Split IRV should return same length as non-split."""
        result_no_split = irv(data)
        result_split = irv(data, split=True, num_split=2)
        assert len(result_no_split) == len(result_split)

    @given(
        arrays(
            dtype=np.float64,
            shape=st.tuples(st.integers(3, 20), st.integers(3, 10)),
            elements=st.floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=30)
    def test_irv_scale_invariance(self, data: np.ndarray) -> None:
        """IRV should scale linearly with data scaling."""
        scale_factor = 2.0
        result_original = irv(data)
        result_scaled = irv(data * scale_factor)
        np.testing.assert_array_almost_equal(result_scaled, result_original * scale_factor)


class TestMahadProperties:
    """Property-based tests for Mahalanobis distance function."""

    @given(valid_survey_data(min_rows=10, max_rows=50, min_cols=2, max_cols=5))
    @settings(max_examples=30)
    def test_mahad_non_negative(self, data: np.ndarray) -> None:
        """Mahalanobis distances should always be non-negative."""
        data = data + np.random.randn(*data.shape) * 0.1
        assume(data.shape[0] > data.shape[1])
        result = mahad(data)
        assert len(result) == data.shape[0]
        assert np.all(result >= 0)

    @given(valid_survey_data(min_rows=20, max_rows=50, min_cols=2, max_cols=5))
    @settings(max_examples=20)
    def test_mahad_flagging_subset(self, data: np.ndarray) -> None:
        """Flagged outliers should be a subset of all observations."""
        data = data + np.random.randn(*data.shape) * 0.1
        assume(data.shape[0] > data.shape[1])
        distances, flags = mahad(data, flag=True)
        assert len(flags) == len(distances)
        assert np.sum(flags) <= len(distances)

    @given(valid_survey_data(min_rows=20, max_rows=50, min_cols=2, max_cols=5))
    @settings(max_examples=20)
    def test_mahad_higher_confidence_fewer_outliers(self, data: np.ndarray) -> None:
        """Higher confidence should flag fewer or equal outliers."""
        data = data + np.random.randn(*data.shape) * 0.1
        assume(data.shape[0] > data.shape[1])
        _, flags_low = mahad(data, flag=True, confidence=0.90)
        _, flags_high = mahad(data, flag=True, confidence=0.99)
        assert np.sum(flags_high) <= np.sum(flags_low)


class TestEvenOddProperties:
    """Property-based tests for even-odd consistency function."""

    @given(
        arrays(
            dtype=np.float64,
            shape=st.tuples(st.integers(2, 20), st.just(12)),
            elements=st.floats(min_value=1, max_value=7, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=30)
    def test_evenodd_output_length(self, data: np.ndarray) -> None:
        """Even-odd should return one score per individual."""
        factors = [6, 6]
        result = evenodd(data, factors)
        assert len(result) == data.shape[0]

    @given(
        arrays(
            dtype=np.float64,
            shape=st.tuples(st.integers(2, 20), st.just(8)),
            elements=st.floats(min_value=1, max_value=7, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=30)
    def test_evenodd_correlation_bounds(self, data: np.ndarray) -> None:
        """Even-odd correlations should be bounded by [-1, 1]."""
        factors = [8]
        result = evenodd(data, factors)
        assert np.all((result >= -1) | np.isnan(result))
        assert np.all((result <= 1) | np.isnan(result))

    @given(st.integers(min_value=2, max_value=10))
    def test_evenodd_perfect_consistency(self, n_individuals: int) -> None:
        """Perfectly consistent responses should have high correlations."""
        data = np.zeros((n_individuals, 8))
        for i in range(n_individuals):
            pattern = np.arange(1, 5)
            data[i, 0::2] = pattern
            data[i, 1::2] = pattern
        factors = [8]
        result = evenodd(data, factors)
        np.testing.assert_array_almost_equal(result, np.ones(n_individuals))


class TestPsychsynProperties:
    """Property-based tests for psychometric synonym function."""

    @given(valid_survey_data(min_rows=5, max_rows=30, min_cols=4, max_cols=10))
    @settings(max_examples=30)
    def test_psychsyn_output_length(self, data: np.ndarray) -> None:
        """Psychsyn should return one score per individual."""
        with np.errstate(invalid="ignore", divide="ignore"):
            result = psychsyn(data, critval=0.3)
        assert len(result) == data.shape[0]

    @given(valid_survey_data(min_rows=5, max_rows=30, min_cols=4, max_cols=10))
    @settings(max_examples=30)
    def test_psychsyn_lower_threshold_more_pairs(self, data: np.ndarray) -> None:
        """Lower critical value should include more or equal item pairs."""
        with np.errstate(invalid="ignore", divide="ignore"):
            _, diag_low = psychsyn(data, critval=0.2, diag=True)
            _, diag_high = psychsyn(data, critval=0.8, diag=True)
        assert np.all(diag_low >= diag_high)

    @given(valid_survey_data(min_rows=5, max_rows=20, min_cols=4, max_cols=8))
    @settings(max_examples=20)
    def test_psychsyn_reproducibility_with_seed(self, data: np.ndarray) -> None:
        """Same seed should produce same results with resampling."""
        with np.errstate(invalid="ignore", divide="ignore"):
            result1 = psychsyn(data, critval=0.3, resample_na=True, random_seed=42)
            result2 = psychsyn(data, critval=0.3, resample_na=True, random_seed=42)
        np.testing.assert_array_equal(result1, result2)


class TestCrossModuleProperties:
    """Property-based tests for cross-module invariants."""

    @given(valid_survey_data(min_rows=20, max_rows=50, min_cols=4, max_cols=8))
    @settings(max_examples=20)
    def test_all_functions_handle_same_data(self, data: np.ndarray) -> None:
        """All main functions should handle the same valid data without errors."""
        assume(data.shape[0] > data.shape[1])

        irv_result = irv(data)
        assert len(irv_result) == data.shape[0]

        perturbed_data = data + np.random.randn(*data.shape) * 0.01
        mahad_result = mahad(perturbed_data)
        assert len(mahad_result) == data.shape[0]

        with np.errstate(invalid="ignore", divide="ignore"):
            psychsyn_result = psychsyn(data, critval=0.3)
        assert len(psychsyn_result) == data.shape[0]

        if data.shape[1] % 2 == 0:
            factors = [data.shape[1]]
            evenodd_result = evenodd(data, factors)
            assert len(evenodd_result) == data.shape[0]


class TestAcquiescenceProperties:
    """Property-based tests for acquiescence index."""

    @given(valid_survey_data())
    @settings(max_examples=50)
    def test_acquiescence_bounded_0_1(self, data: np.ndarray) -> None:
        """Simple mode output should be in [0, 1]."""
        result = acquiescence(data)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)

    @given(valid_survey_data())
    @settings(max_examples=50)
    def test_acquiescence_output_length(self, data: np.ndarray) -> None:
        """Output length should match number of respondents."""
        result = acquiescence(data)
        assert len(result) == data.shape[0]


class TestScreenProperties:
    """Property-based tests for screen function."""

    @given(valid_survey_data(min_rows=5, max_rows=30, min_cols=4, max_cols=10))
    @settings(max_examples=20)
    def test_screen_flag_counts_bounded(self, data: np.ndarray) -> None:
        """Flag counts should be between 0 and n_indices."""
        result = screen(data, indices=["irv", "longstring", "acquiescence"])
        assert np.all(result["flag_counts"] >= 0)
        assert np.all(result["flag_counts"] <= result["n_indices"])
