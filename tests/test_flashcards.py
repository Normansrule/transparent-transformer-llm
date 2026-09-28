"""Every flashcard is complete, and every "learn more" link points at something that exists."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_cards_are_complete_and_links_resolve():
    cards = json.loads((ROOT / "flashcards" / "cards.json").read_text())
    assert len(cards) >= 100
    for c in cards:
        assert c["q"].strip() and c["a"].strip() and c["deck"].strip()
        if c["more"]:
            assert (ROOT / c["more"]).exists(), f"broken link on card: {c['q']}"
    assert len({c["q"] for c in cards}) == len(cards), "duplicate question"
