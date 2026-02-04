"""
Additional helper functions.
"""

import pandas as pd
import json
from urllib.parse import unquote


def urldecode_recursive(s, max_rounds=10):
    """
    Recursively decode URL-encoded strings.

    Args:
        s: String to decode
        max_rounds: Maximum number of decoding rounds to prevent infinite loops

    Returns:
        Decoded string with spaces removed
    """
    if s is None:
        return s
    prev = s
    for _ in range(max_rounds):
        cur = unquote(prev)
        if cur == prev:
            break
        prev = cur
    return prev.replace(" ", "")


def extract_ad_name(s):
    """
    Extract ad name from JSON string.

    Args:
        s: JSON string containing ad information

    Returns:
        Ad name or empty string if extraction fails
    """
    try:
        return json.loads(s)["ad_group_ad"]["ad"]["name"]
    except Exception:
        return ""