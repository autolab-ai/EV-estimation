"""Fast synthetic checks; no competition data, model fit, or scoring."""

import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from route_encoding import add_route_mean_fare, encode_routes, route_cells, route_statistics


class RouteEncodingTests(unittest.TestCase):
    def setUp(self):
        self.forward = [-74.005, 40.705, -73.985, 40.755]
        self.reverse = [-73.985, 40.755, -74.005, 40.705]
        self.unseen = [-73.805, 40.805, -73.705, 40.905]
        self.columns = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]

    def frame(self, trips, index=None):
        return pd.DataFrame(trips, columns=self.columns, index=index)

    def test_direction_smoothing_and_unseen_route(self):
        fitting = self.frame([self.forward, self.forward, self.reverse], index=[7, 4, 18])
        targets = np.array([10.0, 20.0, 40.0])
        cells = route_cells(fitting)
        self.assertEqual(cells.iloc[0].tolist(), [99, 70, 101, 75])
        self.assertNotEqual(tuple(cells.iloc[0]), tuple(cells.iloc[2]))
        stats, mu = route_statistics(cells, targets)
        query = self.frame([self.reverse, self.unseen, self.forward, self.forward])
        actual = encode_routes(route_cells(query), stats, mu)
        expected = [(40 + 50 * mu) / 51, mu, (30 + 50 * mu) / 52, (30 + 50 * mu) / 52]
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
        self.assertEqual(actual.dtype, np.float32)
        outside = self.frame([[-75.005, 39.995, -75.005, 39.995]])
        self.assertTrue((route_cells(outside).to_numpy() == -1).all())

    def test_out_of_fold_alignment_and_no_held_fold_targets(self):
        train = self.frame([self.forward] * 20, index=np.arange(100, 140, 2)[::-1])
        train["original_feature"] = np.arange(20)
        fares = np.arange(1, 21, dtype=np.float64)
        validation = self.frame([self.forward, self.unseen], index=[55, 11])
        validation["original_feature"] = [2, 8]
        test = validation.iloc[::-1].copy()
        encoded_train, encoded_validation, encoded_test = add_route_mean_fare(train, fares, validation, test)
        expected = np.empty(len(train), dtype=np.float32)
        folds = list(KFold(n_splits=5, shuffle=True, random_state=0).split(train))
        for fit_positions, held_positions in folds:
            expected[held_positions] = fares[fit_positions].mean()
        np.testing.assert_allclose(encoded_train["route_mean_fare"], expected)
        np.testing.assert_allclose(encoded_validation["route_mean_fare"], fares.mean())
        np.testing.assert_allclose(encoded_test["route_mean_fare"], fares.mean())
        for original, encoded in [(train, encoded_train), (validation, encoded_validation), (test, encoded_test)]:
            pd.testing.assert_frame_equal(original, encoded[original.columns])
            self.assertEqual(encoded.columns.tolist(), original.columns.tolist() + ["route_mean_fare"])
            self.assertEqual(encoded["route_mean_fare"].dtype, np.float32)
            self.assertNotIn("route_mean_fare", original.columns)
        held_positions = folds[0][1]
        altered_fares = fares.copy()
        altered_fares[held_positions] += 10000.0
        altered, _, _ = add_route_mean_fare(train, altered_fares, validation, test)
        np.testing.assert_array_equal(
            altered["route_mean_fare"].iloc[held_positions],
            encoded_train["route_mean_fare"].iloc[held_positions],
        )


if __name__ == "__main__":
    unittest.main()
