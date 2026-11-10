import pandas as pd

FEATURE_COLUMNS = [
    "hour", "day_of_week", "is_academic",
    "temperature", "humidity",
    "lag_15m", "lag_30m", "lag_60m"
]

def build_features(row):
    values = {
        name: row[name]
        for name in FEATURE_COLUMNS
    }

    assert 0 <= values["hour"] <= 23
    assert 0 <= values["day_of_week"] <= 6
    assert values["is_academic"] in (0, 1)
    assert 24 <= values["temperature"] <= 38
    assert 50 <= values["humidity"] <= 90

    for name in ("lag_15m", "lag_30m", "lag_60m"):
        assert 200 <= values[name] <= 750

    return pd.DataFrame(
        [[values[name] for name in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )