"""
Validación cruzada temporal con embargo (purging).
===================================================

`TimeSeriesSplit` evita usar el futuro para predecir el pasado, pero NO evita la
fuga blanda en la frontera train/val: partidos contiguos (mismo torneo, días
adyacentes) comparten estado ELO/H2H/forma casi idéntico. Si el último partido de
train y el primero de val están a horas, el modelo "ve" información casi del fold
de validación.

El embargo descarta del train las filas a menos de `embargo_days` del inicio del
fold de validación, rompiendo esa contigüidad. El fold de validación queda intacto.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def purged_time_series_splits(dates, n_splits=5, embargo_days=7):
    """
    Genera splits (train_idx, val_idx) tipo TimeSeriesSplit con embargo temporal.

    Parameters
    ----------
    dates : array-like
        Fechas por fila en formato entero yyyymmdd (como `tourney_date`),
        alineadas con X y ordenadas cronológicamente.
    n_splits : int
        Número de folds.
    embargo_days : int
        Días de margen purgados del final del train antes de cada fold de validación.

    Yields
    ------
    (train_idx, val_idx) : tuple of np.ndarray
        El val_idx es idéntico al de TimeSeriesSplit; el train_idx queda purgado.
    """
    dates = np.asarray(dates)
    dts = pd.to_datetime(dates.astype(str), format='%Y%m%d')
    embargo = pd.Timedelta(days=embargo_days)

    for train_idx, val_idx in TimeSeriesSplit(n_splits=n_splits).split(dates):
        cutoff = dts[val_idx].min() - embargo
        keep = dts[train_idx] < cutoff
        yield train_idx[keep], val_idx
