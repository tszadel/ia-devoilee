"""Tests nécessitant des API keys — marqués 'api' pour le CI."""
import pytest, os

pytestmark = pytest.mark.api

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY manquante")
def test_openai_ping():
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Réponds juste 'OK'."}],
        max_tokens=5,
    )
    assert resp.choices[0].message.content is not None

@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY manquante")
def test_anthropic_ping():
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=5,
        messages=[{"role": "user", "content": "Réponds juste 'OK'."}],
    )
    assert resp.content[0].text is not None
