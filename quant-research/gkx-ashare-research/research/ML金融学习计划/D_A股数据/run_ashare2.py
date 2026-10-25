"""Compatibility entry for expanded A-share features. Author: Gilmour Jia.

Uses repaired run_ashare shared rules. Default price19 is deliberate: old
financial caches need --chars 71 --allow-unverified-features and remain exploratory.
"""
from run_ashare import *

if __name__ == '__main__':
    raise SystemExit(main())
