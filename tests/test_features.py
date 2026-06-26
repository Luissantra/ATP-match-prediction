import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.features import FEATURES, RANK_CAP, elo_hibrido, vector_from_features


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
            'diff_elo_general': 100.0, 'diff_elo_sup': 200.0,
            'diff_rank': -5.0, 'is_unranked': 0,
            'diff_age': 2.0,
        }

    def test_orden_coincide_con_FEATURES(self):
        feat = self._feat()
        vec = vector_from_features(feat)
        assert vec == [feat[name] for name in FEATURES]

    def test_longitud_igual_a_FEATURES(self):
        assert len(vector_from_features(self._feat())) == len(FEATURES)

    def test_falta_clave_lanza_error(self):
        feat = self._feat()
        del feat['diff_age']
        with pytest.raises(KeyError):
            vector_from_features(feat)

    def test_features_tiene_5_elementos(self):
        assert len(FEATURES) == 5

    def test_features_no_incluye_h2h_form_level(self):
        for podada in ('diff_h2h', 'diff_form', 'tourney_level_num'):
            assert podada not in FEATURES

    def test_rank_cap_es_250(self):
        assert RANK_CAP == 250
