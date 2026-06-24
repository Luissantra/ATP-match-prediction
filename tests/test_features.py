import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.features import FEATURES, elo_hibrido, vector_from_features


class TestEloHibrido:
    def test_default_5050(self):
        assert elo_hibrido(1600.0, 1400.0) == pytest.approx(1500.0)

    def test_peso_general_total(self):
        assert elo_hibrido(1600.0, 1400.0, w=1.0) == pytest.approx(1600.0)

    def test_peso_superficie_total(self):
        assert elo_hibrido(1600.0, 1400.0, w=0.0) == pytest.approx(1400.0)


class TestVectorFromFeatures:
    def _feat(self):
        return {
            'diff_elo': 50.0, 'diff_rank': -5.0, 'diff_age': 2.0,
            'diff_h2h': 0.3, 'diff_form': 0.1, 'tourney_level_num': 4,
        }

    def test_orden_coincide_con_FEATURES(self):
        feat = self._feat()
        vec = vector_from_features(feat)
        assert vec == [feat[name] for name in FEATURES]

    def test_longitud_igual_a_FEATURES(self):
        assert len(vector_from_features(self._feat())) == len(FEATURES)

    def test_falta_clave_lanza_error(self):
        feat = self._feat()
        del feat['diff_h2h']
        with pytest.raises(KeyError):
            vector_from_features(feat)
