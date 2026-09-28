"""The harness tricks and the parameter formula, tested."""
import sys
from pathlib import Path

from transparent_transformer import GPT, Config
from transparent_transformer.harness import Harness, edit_distance, offline_weather

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))


def test_edit_distance_counts_typing_mistakes():
    assert edit_distance("seatle", "seattle") == 1          # a missing letter
    assert edit_distance("chicgo", "chicago") == 1
    assert edit_distance("teh", "the") == 1                 # swapped neighbours cost one, not two


def test_normalizer_fixes_what_it_should_and_nothing_else():
    h = Harness(offline=True)
    assert h.normalize("whats LA like in summer") == "What is Los Angeles like in summer?"
    assert h.normalize("how is summer in seatle?") == "What is Seattle like in summer?"
    assert "Paris" not in h.normalize("which parts of town are nice")   # short words are never "corrected"


def test_offline_tool_differs_per_city_so_copying_can_be_checked():
    assert offline_weather("Torrance") != offline_weather("Carson")
    assert offline_weather("Torrance") == offline_weather("Torrance")


def test_parameter_formula_matches_the_real_model():
    from make_visuals import n_params
    assert sum(n_params(768, 64, 64, 2, 256).values()) == GPT(Config()).num_parameters() == 153_344
    assert sum(n_params(50257, 1024, 768, 12, 3072).values()) == 124_439_808       # GPT-2 small, as published
