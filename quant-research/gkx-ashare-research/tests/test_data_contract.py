"""Regression tests for economically meaningful timing/return failure modes."""
import importlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "ML金融学习计划" / "D_A股数据"))
from data_contract import (attach_forward_returns, calendar_report_lag, ttm_values,
                           raw_market_cap, require_financial_vintage_opt_in,
                           ensure_derived_directory)
from screen_returns import annotate_returns, add_training_winsor
from build_macro import value_weighted
from build_features import monthly_features, rolling_market_exposure
from build_fundamentals import add_ttm
from build_fundamentals_v2 import monthly_industry_adjustments, add_dividend_flags, _div


class DataTimingTests(unittest.TestCase):
    def test_original_directory_cannot_be_output(self):
        with patch("data_contract.DATA", ROOT), patch("data_contract.DERIVED", ROOT):
            with self.assertRaises(ValueError):
                ensure_derived_directory()

    def test_label_does_not_jump_missing_month(self):
        features = pd.DataFrame({"code": ["a", "a"], "month": ["2020-01", "2020-03"]})
        returns = pd.DataFrame({"code": ["a", "a"], "month": ["2020-03", "2020-04"], "ret": [.8, -.4]})
        result = attach_forward_returns(features, returns)
        self.assertTrue(np.isnan(result.ret_fwd.iloc[0]))
        self.assertEqual(result.ret_fwd.iloc[1], -.4)
        self.assertEqual(result.label_month.iloc[0], "2020-02")

    def test_duplicate_labels_fail(self):
        d = pd.DataFrame({"code": ["a", "a"], "month": ["2020-01"] * 2, "ret": [.1, .2]})
        with self.assertRaises(ValueError):
            attach_forward_returns(d.iloc[:1], d)

    def test_eligibility_preserves_future_extreme_returns(self):
        d = pd.DataFrame({"code": ["a"] * 3, "month": ["2020-01", "2020-02", "2020-03"], "ret": [.1, 7.45, -.8]})
        result = annotate_returns(d, {("a", "2020-01"): 20, ("a", "2020-02"): 1, ("a", "2020-03"): 20})
        np.testing.assert_array_equal(result.ret, d.ret)
        self.assertEqual(result.eligible_asof.tolist(), [False, False, True])

    def test_winsor_training_threshold_immune_to_test_change(self):
        d = pd.DataFrame({"month": ["2018-01", "2018-02", "2022-01"], "ret": [-.3, .5, 10.]})
        first, bounds1 = add_training_winsor(d, .1, "2018-01", "2018-12")
        d.loc[2, "ret"] = 10000
        second, bounds2 = add_training_winsor(d, .1, "2018-01", "2018-12")
        self.assertEqual(bounds1, bounds2)
        self.assertTrue(np.isnan(second.ret_train.iloc[2]))
        self.assertEqual(first.ret.iloc[2], 10.)

    def test_exact_quarter_lag(self):
        d = pd.DataFrame({"SECURITY_CODE": ["a"] * 5,
                          "REPORT_DATE": pd.to_datetime(["2019-03-31", "2019-06-30", "2019-12-31", "2020-03-31", "2020-06-30"]),
                          "v": [1., 2., 4., 5., 6.]})
        result = calendar_report_lag(d, d.v)
        self.assertEqual(result.iloc[3], 1.)
        self.assertEqual(result.iloc[4], 2.)
        self.assertTrue(np.isnan(result.iloc[2]))

    def test_ttm_missing_history_not_replaced_with_ytd(self):
        d = pd.DataFrame({"code": ["a"], "REPORT_DATE": pd.to_datetime(["2020-03-31"]),
                          "NOTICE_DATE": pd.to_datetime(["2020-04-20"]), "PARENT_NETPROFIT": [20.]})
        self.assertTrue(np.isnan(add_ttm(d).PARENT_NETPROFIT_TTM.iloc[0]))

    def test_ttm_waits_for_all_input_publications(self):
        d = pd.DataFrame({"SECURITY_CODE": ["a"] * 3,
                          "REPORT_DATE": pd.to_datetime(["2020-03-31", "2020-12-31", "2021-03-31"]),
                          "NOTICE_DATE": pd.to_datetime(["2020-04-20", "2021-05-01", "2021-04-20"]),
                          "v": [20., 100., 30.]})
        vals, dates = ttm_values(d, "v")
        self.assertEqual(vals.iloc[2], 110.)
        self.assertEqual(dates.iloc[2], pd.Timestamp("2021-05-01"))

    def test_vintage_guard(self):
        with self.assertRaises(SystemExit):
            require_financial_vintage_opt_in(False)
        require_financial_vintage_opt_in(True)

    def test_missing_value_does_not_dilute_market_ratio(self):
        d = pd.DataFrame({"month": ["2020-01"] * 2, "mc": [1., 9.], "x": [.1, np.nan]})
        self.assertAlmostEqual(value_weighted(d, "x").iloc[0], .1)

    def test_market_cap_requires_raw_column(self):
        with self.assertRaises(ValueError):
            raw_market_cap(pd.DataFrame({"mvel1": [1.]}))

    def test_daily_features_remove_ex_dividend_price_jump(self):
        px = pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
                           "open": [10., 5., 5.1], "high": [10., 5., 5.1], "low": [10., 5., 5.1],
                           "close": [10., 5., 5.1], "volume": [100., 200., 200.]})
        sh = pd.DataFrame({"SECURITY_CODE": ["600000"], "date": pd.to_datetime(["2019-12-31"]),
                           "TOTAL_SHARES": [1000.], "LISTED_A_SHARES": [1000.]})
        div = pd.DataFrame({"SECURITY_CODE": ["600000"], "date": pd.to_datetime(["2020-01-03"]),
                            "cash_per_share": [0.], "bonus": [1.]})
        out = monthly_features(px, sh, div, "sh600000", pd.Series({"2020-01": 3}))
        self.assertAlmostEqual(out.retvol.iloc[0], np.std([0., .02], ddof=1))
        self.assertAlmostEqual(out.market_cap.iloc[0], 5100.)

    def test_beta_matches_exact_linear_model(self):
        market = np.linspace(-.1, .1, 36)
        beta, idio = rolling_market_exposure(.03 + 2 * market, market)
        np.testing.assert_allclose(beta[23:], 2.)
        np.testing.assert_allclose(idio[23:], 0., atol=1e-15)

    def test_future_dividend_flags_disabled(self):
        out = add_dividend_flags(pd.DataFrame({"x": [1, 2]}))
        self.assertTrue(out[["divi", "divo"]].isna().all().all())

    def test_industry_aggregation_cannot_use_future_month(self):
        out = pd.DataFrame({"code6": ["a", "b", "a", "b"],
                            "month": ["2020-01"] * 2 + ["2020-02"] * 2,
                            "TOTAL_OPERATE_INCOME": [10., 30., 100000., 1.]})
        cols = ["pchcapx_ia", "chatoia", "chpmia", "chempia", "bm_ia", "cfp_ia", "mve_ia"]
        for col in cols:
            out[col] = [1., 3., 5000., 1.]
        ind = out[["code6", "month"]].assign(industry_code="i")
        result = monthly_industry_adjustments(out, ind)
        np.testing.assert_allclose(result.bm_ia.iloc[:2], [-1., 1.])
        np.testing.assert_allclose(result.herf.iloc[:2], [.625, .625])

    def test_missing_optional_denominator(self):
        self.assertTrue(_div(pd.Series([1., 2.]), None).isna().all())


if __name__ == "__main__":
    unittest.main()
