"""Tests for shared matrix input validation."""

import unittest

import numpy as np

from ier._validation import validate_matrix_input


class TestValidateMatrixInput(unittest.TestCase):
    def test_matching_numpy_dtype_is_reused(self) -> None:
        data = np.arange(12, dtype=float).reshape(3, 4)

        result = validate_matrix_input(data, dtype=float)

        self.assertIs(result, data)

    def test_mismatched_numpy_dtype_is_converted(self) -> None:
        data = np.arange(12, dtype=np.int64).reshape(3, 4)

        result = validate_matrix_input(data, dtype=float)

        self.assertIsNot(result, data)
        self.assertEqual(result.dtype, np.dtype(float))
        np.testing.assert_array_equal(result, data)


if __name__ == "__main__":
    unittest.main()
