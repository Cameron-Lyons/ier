"""Tests for public index registry discovery."""

import unittest

from ier import index_catalog


class TestIndexCatalog(unittest.TestCase):
    def test_catalog_describes_all_registered_indices(self) -> None:
        catalog = index_catalog()

        self.assertEqual(len(catalog), 21)
        self.assertEqual(
            catalog["irv"],
            {
                "flag_direction": "low",
                "flag_mode": "percentile",
                "default_screen": True,
                "default_composite": True,
                "composite_enabled": True,
                "required_options": (),
            },
        )
        self.assertEqual(catalog["onset"]["flag_mode"], "present")
        self.assertFalse(catalog["onset"]["composite_enabled"])
        self.assertEqual(
            catalog["missing_rate"],
            {
                "flag_direction": "high",
                "flag_mode": "percentile",
                "default_screen": False,
                "default_composite": False,
                "composite_enabled": True,
                "required_options": (),
            },
        )
        self.assertEqual(catalog["evenodd"]["required_options"], ("evenodd_factors",))
        self.assertEqual(
            catalog["infrequency"]["required_options"],
            ("infrequency_item_indices", "infrequency_expected_responses"),
        )

    def test_catalog_returns_independent_metadata(self) -> None:
        catalog = index_catalog()
        catalog["irv"]["default_screen"] = False

        self.assertTrue(index_catalog()["irv"]["default_screen"])
