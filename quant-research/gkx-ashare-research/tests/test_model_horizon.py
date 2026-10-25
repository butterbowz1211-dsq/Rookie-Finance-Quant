"""Ensure multi-horizon Monte Carlo labels do not overlap across splits."""
import pathlib
import sys
import unittest
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research"/"ML金融学习计划"/"A_蒙特卡洛"))
from run_mc import build_design, split_three


class HorizonTests(unittest.TestCase):
    def test_monthly_split_unchanged(self):
        tr,va,te=split_three(180,horizon=1,n_stocks=2)
        np.testing.assert_array_equal(tr,np.arange(120))
        np.testing.assert_array_equal(va,np.arange(120,240))
        np.testing.assert_array_equal(te,np.arange(240,360))

    def test_all_response_shocks_disjoint_across_splits(self):
        for horizon in (3,6,12):
            total=180-horizon+1
            tr,va,te=split_three(total,horizon=horizon,n_stocks=2)
            def used_shocks(index):
                return {t+k for t in np.unique(index//2) for k in range(horizon)}
            a,b,c=map(used_shocks,(tr,va,te))
            self.assertFalse(a&b)
            self.assertFalse(b&c)
            self.assertLess(max(a),min(b))
            self.assertLess(max(b),min(c))

    def test_summed_response_uses_correct_horizon(self):
        state={"c":np.ones((12,2,3)),"x":np.arange(12.),"r":np.arange(24.).reshape(12,2)}
        x,y,t=build_design(state,3)
        self.assertEqual(x.shape,(20,6))
        np.testing.assert_array_equal(y.reshape(t,2)[0],state["r"][:3].sum(axis=0))
        with self.assertRaises(ValueError): build_design(state,0)
        with self.assertRaises(ValueError): split_three(10,12,2)


if __name__=="__main__": unittest.main()
