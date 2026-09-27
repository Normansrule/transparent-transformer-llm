"""Exercise 08 - Alignment.   Lesson: stages/08_alignment/   Check your work:  python classroom/check.py"""
import numpy as np  # noqa: F401


def dpo_loss(logp_chosen, logp_rejected, ref_chosen, ref_rejected, beta=0.2):
    """Direct Preference Optimization (DPO) loss for ONE pair of answers (all inputs are floats: log-probabilities).
    z = beta * ((logp_chosen - logp_rejected) - (ref_chosen - ref_rejected));   loss = -log(sigmoid(z))"""
    raise NotImplementedError("your code here")
