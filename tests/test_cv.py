import sys
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.cv import purged_time_series_splits


def _fechas_int(n=100):
    """n días consecutivos como enteros yyyymmdd (formato de tourney_date)."""
    rng = pd.date_range('2024-01-01', periods=n, freq='D')
    return rng.strftime('%Y%m%d').astype(int).to_numpy()


def test_genera_n_splits():
    splits = list(purged_time_series_splits(_fechas_int(), n_splits=5))
    assert len(splits) == 5


def test_validacion_identica_a_timeseriessplit():
    dates = _fechas_int()
    purged = list(purged_time_series_splits(dates, n_splits=5, embargo_days=7))
    base = list(TimeSeriesSplit(n_splits=5).split(dates))
    for (_, pv), (_, bv) in zip(purged, base):
        assert np.array_equal(pv, bv)  # el embargo no toca el fold de validación


def test_train_purgado_es_subconjunto_del_base():
    dates = _fechas_int()
    purged = list(purged_time_series_splits(dates, n_splits=5, embargo_days=7))
    base = list(TimeSeriesSplit(n_splits=5).split(dates))
    for (pt, _), (bt, _) in zip(purged, base):
        assert set(pt.tolist()).issubset(set(bt.tolist()))


def test_ninguna_fila_train_dentro_del_embargo():
    dates = _fechas_int()
    embargo_days = 7
    for train_idx, val_idx in purged_time_series_splits(dates, n_splits=5, embargo_days=embargo_days):
        val_start = pd.to_datetime(str(dates[val_idx].min()))
        train_dts = pd.to_datetime(dates[train_idx].astype(str))
        cutoff = val_start - pd.Timedelta(days=embargo_days)
        assert (train_dts < cutoff).all()


def test_embargo_cero_equivale_a_timeseriessplit():
    dates = _fechas_int()
    purged = list(purged_time_series_splits(dates, n_splits=5, embargo_days=0))
    base = list(TimeSeriesSplit(n_splits=5).split(dates))
    for (pt, _), (bt, _) in zip(purged, base):
        assert np.array_equal(pt, bt)
