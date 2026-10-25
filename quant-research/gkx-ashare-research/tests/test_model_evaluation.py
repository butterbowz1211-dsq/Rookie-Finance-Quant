"""Targeted regression tests for observed research bugs. Author: Gilmour Jia."""
import pathlib
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "ML金融学习计划" / "A_蒙特卡洛"))
sys.path.insert(0, str(ROOT / "research" / "ML金融学习计划" / "D_A股数据"))
import evaluation as ev
import models
import nn
from sklearn.preprocessing import SplineTransformer
from portfolio_check import decile_portfolios
from run_ashare import _merge


class EvaluationTests(unittest.TestCase):
    def test_metric_benchmarks_are_distinct(self):
        y = np.array([1., 3.])
        p = np.array([1., 1.])
        self.assertAlmostEqual(ev.r2_zero(y,p), .6)
        self.assertAlmostEqual(ev.r2_sample_mean(y,p), -1)
        self.assertAlmostEqual(ev.r2_training_mean(y,p,1), 0)
        with self.assertRaises(ValueError):
            ev.r2_zero(y, np.array([np.nan,1]))
        with self.assertRaises(ValueError):
            ev.r2_zero(y, np.array([[1., 1.]]))
        with self.assertRaises(ValueError):
            ev.r2_zero(y, np.array([1.]))

    def test_duplicate_join_rejected(self):
        left = pd.DataFrame({"code":["A"], "month":["2020-01"]})
        right = pd.DataFrame({"code":["A","A"], "month":["2020-01","2020-01"], "x":[1,2]})
        with self.assertRaises(pd.errors.MergeError): _merge(left,right)
        with self.assertRaises(ValueError):
            decile_portfolios(pd.DataFrame({"code":["A"]}), "missing")

    def test_target_month_boundary(self):
        df = pd.DataFrame({"month":["2018-11","2018-12","2021-12"],
                           "label_month":["2018-12","2019-01","2022-01"]})
        masks = ev.split_masks(df)
        np.testing.assert_array_equal(masks["train"],[True,False,False])
        np.testing.assert_array_equal(masks["valid"],[False,True,False])
        np.testing.assert_array_equal(masks["test"],[False,False,True])
        df.loc[1,"label_month"] = "2019-02"
        with self.assertRaises(ValueError): ev.split_masks(df)

    def test_rank_equation_and_future_missingness(self):
        df = pd.DataFrame({"month":["2020-01"]*4,"x":[1.,2.,3.,np.nan],
                           "ret_fwd":[1.,np.nan,2.,2.]})
        ranked = ev.rank_characteristics(df,["x"])
        np.testing.assert_allclose(ranked.x,[-.5,0,.5,0])
        self.assertEqual(ranked.loc[0,"x"], -.5)

    def test_industry_vocabulary_only_uses_training(self):
        train = [True,True,True,False,False]
        a, config = ev.encode_industry(pd.Series(["A","A","B","C","C"]),train,1)
        b, config2 = ev.encode_industry(pd.Series(["A","A","B","D","E"]),train,1)
        self.assertEqual(config,config2)
        self.assertEqual(list(a),list(b))
        np.testing.assert_array_equal(a.iloc[:3].to_numpy(),b.iloc[:3].to_numpy())

    def test_initial_drawdown_and_geometric_return(self):
        result = ev.portfolio_statistics([-.2,.25])
        self.assertAlmostEqual(result["max_drawdown"],-.2)
        self.assertAlmostEqual(result["cagr"],0)

    def test_constant_predictions_cannot_form_deciles(self):
        df = pd.DataFrame({"code":range(30),"month":"2020-01","label_month":"2020-02",
                           "ret_fwd":np.linspace(-.1,.1,30),"p":0.})
        self.assertTrue(decile_portfolios(df,"p").empty)

    def test_rank_deficient_pcr(self):
        x = np.arange(30.).reshape(-1,1)
        x = np.column_stack([x,x,np.ones(30)])
        predict,info = models.fit_pcr(x[:20],x[:20,0],range(1,4),x[20:],x[20:,0])
        self.assertTrue(np.isfinite(predict(x)).all())
        np.testing.assert_allclose(predict(x),x[:,0],atol=1e-10)
        self.assertEqual(info["k"],1)

    def test_tree_parameters_and_seed_reproducibility(self):
        rng=np.random.default_rng(3); x=rng.normal(size=(100,3)); y=x[:,0]**2
        for fit,grid in [(models.fit_rf,[(2,2)]),(models.fit_gbrt,[(2,.1,8)])]:
            kwargs={"n_trees":8} if fit is models.fit_rf else {}
            p,info=fit(x[:70],y[:70],grid,x[70:],y[70:],seed=9,**kwargs)
            q,_=fit(x[:70],y[:70],grid,x[70:],y[70:],seed=9,**kwargs)
            self.assertIsNotNone(info["best"])
            np.testing.assert_allclose(p(x),q(x))

    def test_huber_glm_matches_weighted_least_squares_for_one_group(self):
        rng = np.random.default_rng(31)
        x = np.linspace(-2,2,160).reshape(-1,1)
        y = x[:,0]**2 + rng.normal(scale=.1,size=len(x))
        y[15] += 20
        spl = SplineTransformer(n_knots=3,degree=3,include_bias=False,extrapolation="linear")
        basis = spl.fit_transform(x)
        q,_ = np.linalg.qr(basis-basis.mean(axis=0))
        q *= np.sqrt(len(x))
        centered = y-y.mean()
        beta = np.linalg.lstsq(q,centered,rcond=None)[0]
        for _ in range(8):
            residual = centered-q@beta
            xi = np.quantile(np.abs(residual),.999)
            weight = np.minimum(1.,xi/np.maximum(np.abs(residual),1e-12))
            sw = np.sqrt(weight)
            beta = np.linalg.lstsq(q*sw[:,None],centered*sw,rcond=None)[0]
        predict,_ = models.fit_glm(x,y,[0.],x,y,huber=True)
        np.testing.assert_allclose(predict(x),q@beta+y.mean(),atol=1e-6)

    def test_nn_reproducible_and_records_training(self):
        rng=np.random.default_rng(4); x=rng.normal(size=(80,3)); y=x[:,0]*.01
        p,info=nn.fit_nn(x[:60],y[:60],x[60:],y[60:],1,[(1e-4,.01)],ensemble=2,epochs=2,seed=5)
        q,_=nn.fit_nn(x[:60],y[:60],x[60:],y[60:],1,[(1e-4,.01)],ensemble=2,epochs=2,seed=5)
        np.testing.assert_allclose(p(x),q(x))
        self.assertEqual(info["ensemble_seeds"],[5100,5101])
        self.assertEqual(len(info["best_epochs"]),2)


if __name__ == "__main__": unittest.main()
