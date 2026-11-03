import unittest
import pandas as pd
from src.analysis import prepare_events, experiment_result, sample_ratio_mismatch, by_city

class ExperimentTests(unittest.TestCase):
    def setUp(self):
        events = pd.DataFrame({
            "id_view": list(range(1, 9)),
            "time_view": ["2023-01-01 12:00:00"] * 8,
            "id_group": [0,0,0,0,1,1,1,1],
            "id_city": [1] * 8,
            "nflag_order": [0,0,1,1,1,1,1,0],
        })
        cities = pd.DataFrame({"id_city":[1], "name_city":["Демо"]})
        self.df = prepare_events(events, cities)

    def test_effect(self):
        r = experiment_result(self.df)
        self.assertAlmostEqual(r["effect_pp"], 25.)
        self.assertEqual(r["a_n"], 4)
        self.assertEqual(r["b_n"], 4)
        self.assertLessEqual(r["ci_low_pp"], r["effect_pp"])
        self.assertGreaterEqual(r["ci_high_pp"], r["effect_pp"])

    def test_srm(self):
        self.assertAlmostEqual(sample_ratio_mismatch(self.df)["p_value"], 1.)

    def test_city(self):
        self.assertEqual(by_city(self.df).iloc[0]["city"], "Демо")

    def test_duplicate_detected(self):
        copy = self.df.drop(columns=["name_city"]).copy()
        copy.loc[1, "id_view"] = 1
        with self.assertRaises(ValueError):
            prepare_events(copy, pd.DataFrame({"id_city":[1],"name_city":["Демо"]}))

if __name__ == "__main__":
    unittest.main()
