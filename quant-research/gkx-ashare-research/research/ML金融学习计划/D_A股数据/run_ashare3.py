"""Compatibility entry for full-statement features. Author: Gilmour Jia.

Run with --chars fund56 --allow-unverified-features only after acknowledging
financial vintage limitations; default uses repaired price19 features.
"""
from run_ashare import *

if __name__ == '__main__':
    raise SystemExit(main())
