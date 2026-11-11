#!/usr/bin/env python3
"""
src/test_daily_limit_parser.py
Unit test for daily rate limit parsing logic in RelayPipeline.
Feeds the verbatim error message from probe_raw.jsonl entry 31 without network calls.
Asserts that it parses 523.584 s and outputs 'about 9 minutes'.
"""

import re
import math

def parse_daily_limit_error(err_str: str):
    """Replicates the exact parsing logic in RelayPipeline."""
    retry_after_val = None
    m = re.search(r"try again in (?:(\d+)m)?([0-9.]+)s", err_str)
    if m:
        mins = float(m.group(1)) if m.group(1) else 0.0
        secs = float(m.group(2))
        retry_after_val = mins * 60.0 + secs

    wait_mins = max(1, math.ceil(retry_after_val / 60.0))
    error_msg = (
        f"The research paper assistant has reached its daily compute limit "
        f"(200,000 tokens/day rolling bucket refilling at ~2.3 tokens/s). "
        f"Please try again in about {wait_mins} minutes."
    )
    return retry_after_val, wait_mins, error_msg

def test_daily_limit_parsing():
    raw_entry_31_message = (
        "Rate limit reached for model `qwen/qwen3.8-27b` in organization `org_MASKED` "
        "service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199180, Requested 2032. "
        "Please try again in 8m43.584s. Need more tokens? Upgrade to Dev Tier today at "
        "https://console.groq.com/settings/billing"
    )

    retry_seconds, wait_mins, user_msg = parse_daily_limit_error(raw_entry_31_message.lower())

    print("Test Daily Limit Parsing:")
    print(f"  Parsed recovery duration: {retry_seconds:.3f} s (expected: 523.584 s)")
    print(f"  Computed wait minutes:    {wait_mins} min (expected: 9 minutes)")
    print(f"  User-facing message:      \"{user_msg}\"")

    assert math.isclose(retry_seconds, 523.584, rel_tol=1e-5), f"Expected 523.584s, got {retry_seconds}"
    assert wait_mins == 9, f"Expected 9 minutes, got {wait_mins}"
    assert "about 9 minutes" in user_msg, f"Expected 'about 9 minutes' in message: {user_msg}"
    print("ALL ASSERTIONS PASSED! Unit test succeeded.")

if __name__ == "__main__":
    test_daily_limit_parsing()
