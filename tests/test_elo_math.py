import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.elo import calcular_expectativa, actualizar_ratings, _extraer_sets, _mov_factor, _k_for_player


class TestCalcularExpectativa:
    def test_ratings_iguales_da_50_pct(self):
        assert calcular_expectativa(1500, 1500) == pytest.approx(0.5, abs=1e-9)

    def test_diferencia_400_da_aprox_91_pct(self):
        # E_A = 1 / (1 + 10^(-400/400)) = 1 / (1 + 10^-1) = 10/11 ≈ 0.909
        result = calcular_expectativa(1900, 1500)
        assert result == pytest.approx(10 / 11, abs=1e-6)

    def test_diferencia_negativa_400_da_aprox_9_pct(self):
        result = calcular_expectativa(1500, 1900)
        assert result == pytest.approx(1 / 11, abs=1e-6)

    def test_expectativa_entre_0_y_1(self):
        for diff in [-800, -400, -200, 0, 200, 400, 800]:
            e = calcular_expectativa(1500 + diff, 1500)
            assert 0.0 < e < 1.0

    def test_expectativas_complementarias(self):
        e_a = calcular_expectativa(1600, 1400)
        e_b = calcular_expectativa(1400, 1600)
        assert e_a + e_b == pytest.approx(1.0, abs=1e-9)


class TestActualizarRatings:
    def test_suma_cero_ganador_favorito(self):
        nuevo_g, nuevo_p = actualizar_ratings(1600, 1400, resultado_A=1)
        cambio_g = nuevo_g - 1600
        cambio_p = nuevo_p - 1400
        assert cambio_g + cambio_p == pytest.approx(0.0, abs=0.2)

    def test_suma_cero_upset(self):
        nuevo_g, nuevo_p = actualizar_ratings(1400, 1600, resultado_A=1)
        cambio_g = nuevo_g - 1400
        cambio_p = nuevo_p - 1600
        assert cambio_g + cambio_p == pytest.approx(0.0, abs=0.2)

    def test_upset_da_mas_puntos_que_victoria_esperada(self):
        # Favorito (1600) pierde ante underdog (1400)
        _, nuevo_underdog_upset = actualizar_ratings(1400, 1600, resultado_A=1)
        ganancia_upset = nuevo_underdog_upset - 1400

        # Favorito (1600) gana a underdog (1400) — resultado esperado
        nuevo_fav_esperado, _ = actualizar_ratings(1600, 1400, resultado_A=1)
        ganancia_esperada = nuevo_fav_esperado - 1600

        assert ganancia_upset > ganancia_esperada

    def test_k_cero_no_cambia_ratings(self):
        nuevo_g, nuevo_p = actualizar_ratings(1500, 1500, resultado_A=1, K=0)
        assert nuevo_g == pytest.approx(1500.0, abs=0.1)
        assert nuevo_p == pytest.approx(1500.0, abs=0.1)

    def test_ganador_sube_perdedor_baja(self):
        nuevo_g, nuevo_p = actualizar_ratings(1500, 1500, resultado_A=1)
        assert nuevo_g > 1500
        assert nuevo_p < 1500

    def test_retorna_dos_floats(self):
        result = actualizar_ratings(1500, 1500, resultado_A=1)
        assert len(result) == 2
        assert all(isinstance(r, float) for r in result)

    def test_no_redondea_internamente(self):
        # Con ratings distintos, el delta tiene muchos decimales.
        # actualizar_ratings NO debe redondear: debe devolver full precision.
        a, b = 1500.0, 1520.0
        nuevo_a, _ = actualizar_ratings(a, b, resultado_A=1)
        e_a = 1 / (1 + 10 ** ((b - a) / 400))
        delta = 32 * (1 - e_a)
        assert nuevo_a == pytest.approx(a + delta, abs=1e-9)


# _extraer_sets
def test_extraer_sets_bo3_straight():
    assert _extraer_sets("6-4 6-2") == (2, 0)

def test_extraer_sets_bo3_deciding():
    assert _extraer_sets("4-6 7-5 6-3") == (2, 1)

def test_extraer_sets_bo5_straight():
    assert _extraer_sets("6-3 6-2 6-1") == (3, 0)

def test_extraer_sets_bo5_one_loss():
    assert _extraer_sets("6-3 4-6 6-2 7-5") == (3, 1)

def test_extraer_sets_bo5_deciding():
    assert _extraer_sets("6-3 4-6 6-2 4-6 6-3") == (3, 2)

def test_extraer_sets_tiebreak_notation():
    assert _extraer_sets("7-6(5) 6-4") == (2, 0)

def test_extraer_sets_ret():
    assert _extraer_sets("6-3 4-6 RET") == (0, 0)

def test_extraer_sets_empty():
    assert _extraer_sets("") == (0, 0)

def test_extraer_sets_wo():
    assert _extraer_sets("W/O") == (0, 0)

# _mov_factor
def test_mov_factor_straight_bo3():
    assert _mov_factor(2, 0) == pytest.approx(1.25)

def test_mov_factor_deciding_bo3():
    assert _mov_factor(2, 1) == pytest.approx(1.0)

def test_mov_factor_bo5_straight():
    assert _mov_factor(3, 0) == pytest.approx(1.5)

def test_mov_factor_bo5_one_loss():
    assert _mov_factor(3, 1) == pytest.approx(1.25)

def test_mov_factor_bo5_deciding():
    assert _mov_factor(3, 2) == pytest.approx(1.0)

def test_mov_factor_unparseable_gives_one():
    assert _mov_factor(0, 0) == pytest.approx(1.0)

# _k_for_player
def test_k_provisional_for_debutantes():
    assert _k_for_player(0) == 48
    assert _k_for_player(9) == 48

def test_k_intermediate():
    assert _k_for_player(10) == 40
    assert _k_for_player(29) == 40

def test_k_established():
    assert _k_for_player(30) == 32
    assert _k_for_player(1000) == 32
