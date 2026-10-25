"""Render aggregate research figures. Author: Gilmour Jia.

Uses only public aggregate CSVs; no individual stock data is required.
"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results' / 'repaired' / 'monthly_summary'
OUT = ROOT / 'docs' / 'figures'

def main():
    with (SOURCE / 'pooled_metrics_recomputed.csv').open(encoding='utf-8') as stream:
        rows = {row['model']: row for row in csv.DictReader(stream)}
    order = ['ols', 'ridge', 'lasso', 'enet', 'pcr', 'pls', 'rf', 'gbrt', 'nn3']
    labels = ['OLS', 'Ridge', 'Lasso', 'ENet', 'PCR', 'PLS', 'RF', 'GBRT', 'NN3']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    y = np.arange(len(order))
    values = [float(rows[m]['pooled_r2_zero_percent']) for m in order]
    axes[0].barh(y, values, color='#287D8E')
    for i, value in enumerate(values):
        axes[0].text(value + .018, i, f'{value:.3f}%', va='center', fontsize=9)
    axes[0].set(yticks=y, yticklabels=labels, xlim=(0, 1.4),
                xlabel='Pooled zero-benchmark R² (%)', title='A. Prediction error')
    ic = [float(rows[m]['mean_monthly_spearman_rank_ic']) for m in order]
    colors = ['#BA6A25' if m == 'gbrt' else '#287D8E' for m in order]
    axes[1].barh(y, ic, color=colors)
    for i, model in enumerate(order):
        count = int(rows[model]['spearman_rank_ic_valid_months'])
        x = ic[i] + .004 if ic[i] >= 0 else ic[i] - .004
        axes[1].text(x, i, f'{ic[i]:.3f} ({count}/48)', va='center',
                     ha='left' if ic[i] >= 0 else 'right', fontsize=9)
    axes[1].axvline(0, color='#666666', lw=.8)
    axes[1].set(yticks=y, yticklabels=labels, xlim=(-.2, .23),
                xlabel='Mean monthly Rank IC (valid months / 48)',
                title='B. Cross-sectional ranking')
    for axis in axes:
        axis.invert_yaxis()
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='x', alpha=.15)
        axis.set_axisbelow(True)
    fig.suptitle('Repaired price19 experiment | target months 2022–2025\n'
                 'Total returns; fixed split; descriptive results without confidence intervals',
                 fontsize=12)
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ['svg', 'png']:
        fig.savefig(OUT / f'repaired_model_comparison.{extension}', dpi=180)
    plt.close(fig)

if __name__ == '__main__':
    main()
