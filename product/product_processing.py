"""
Product list processing and parsing functions.
"""

import pandas as pd
import numpy as np
import ast


def parse_collections(val):
    if pd.isna(val):
        return []

    if isinstance(val, list):
        return val

    s = str(val).strip()
    if not s:
        return []

    # Must be bracketed to be treated as list
    if s.startswith("[") and s.endswith("]"):
        # Try Python literal parsing first: handles "['A', 'B']"
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            # Fall back to manual split for "[A, B, C]" style
            inner = s[1:-1].strip()
            if not inner:
                return []
            return [x.strip(" '\"") for x in inner.split(",") if x.strip(" '\"")]

    # Not a list, treat the whole thing as one label
    return [s]


def ensure_list(x):
    if x is None or x is pd.NA or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, (tuple, set)):
        return list(x)
    if hasattr(x, "tolist"):  # numpy array
        try:
            return x.tolist()
        except Exception:
            pass
    return [x]


def to_scalar(x):
    if x is None or x is pd.NA or (isinstance(x, float) and pd.isna(x)):
        return pd.NA
    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            return to_scalar(x[0])
        return " | ".join(str(i) for i in x)
    if hasattr(x, "tolist"):
        try:
            return to_scalar(x.tolist())
        except Exception:
            return str(x)
    return str(x)