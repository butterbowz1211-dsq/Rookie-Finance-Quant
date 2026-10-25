"""Independent arithmetic checks for publicly exported aggregates."""
import pathlib
import sys
import unittest
import numpy as np
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from summarize_predictions import join_predictions,summarize,check_reported,correlation


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.frame=pd.DataFrame({"code":["A","B","A","B"],
            "month":["2020-01","2020-01","2020-02","2020-02"],
            "label_month":["2020-02","2020-02","2020-03","2020-03"],
            "market_cap":[1.,2.,1.,2.],"ret_fwd":[1.,3.,2.,4.],"a":[1.,1.,2.,2.]})

    def test_strict_join_rejects_wrong_y_or_dropped_key(self):
        other=self.frame.rename(columns={"a":"b"}).copy()
        joined,names=join_predictions([self.frame,other])
        self.assertEqual(names,["a","b"])
        self.assertEqual(len(joined),4)
        other.loc[0,"ret_fwd"]+=.001
        with self.assertRaises(ValueError): join_predictions([self.frame,other])
        with self.assertRaises(ValueError): join_predictions([self.frame,other.iloc[1:]])
        with self.assertRaises(pd.errors.MergeError): join_predictions([self.frame,pd.concat([other,other.iloc[:1]])])

    def test_pooled_is_ratio_of_sums_not_mean_of_ratios(self):
        joined,names=join_predictions([self.frame])
        monthly,pooled=summarize(joined,names)
        self.assertAlmostEqual(pooled.iloc[0].pooled_r2_zero,1-8/30)
        self.assertAlmostEqual(monthly.r2_zero.mean(),.7)
        self.assertTrue(monthly.pearson_ic.isna().all())
        check_reported(pooled,[pd.DataFrame({"model":["a"],"OOS":[100*(1-8/30)]})])
        with self.assertRaises(ValueError):
            check_reported(pooled,[pd.DataFrame({"model":["a"],"OOS":[70.]})])

    def test_identical_decimal_forecast_has_undefined_correlation(self):
        # This exact constant can have a nonzero std from floating point rounding.
        constant=np.full(5000,0.0011137123456789)
        actual=np.arange(5000,dtype=float)
        self.assertTrue(np.isnan(correlation(constant,actual)))
        self.assertTrue(np.isnan(correlation(actual,constant)))
        self.assertAlmostEqual(correlation(actual,actual),1.)

    def test_effective_correlation_months_are_explicit(self):
        frame=self.frame.copy()
        frame.loc[3,"a"]=3.
        monthly,pooled=summarize(frame,["a"])
        self.assertEqual(pooled.iloc[0].months,2)
        self.assertEqual(pooled.iloc[0].pearson_ic_valid_months,1)
        self.assertEqual(pooled.iloc[0].spearman_rank_ic_valid_months,1)
        self.assertAlmostEqual(pooled.iloc[0].mean_monthly_pearson_ic,1.)


if __name__=="__main__": unittest.main()
