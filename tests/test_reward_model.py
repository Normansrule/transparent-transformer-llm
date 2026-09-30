"""The hand-written Bradley-Terry reward model learns a preference it can see."""
import numpy as np

from transparent_transformer.reward_model import RewardModel


def test_bradley_terry_learns_a_visible_preference():
    rng = np.random.default_rng(0)
    chosen = rng.normal(size=(200, 6)) + np.array([3.0, 0, 0, 0, 0, 0])      # better answers are higher on feature 0 (a perfect scorer gets ~98%)
    rejected = rng.normal(size=(200, 6))
    rm = RewardModel().fit(chosen[:150], rejected[:150])
    assert rm.accuracy(chosen[150:], rejected[150:]) > 0.9
    assert rm.w[0] > abs(rm.w[1:]).max()                                   # it found the feature that matters
