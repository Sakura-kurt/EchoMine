"""
Unit tests for CharacterResponse schema and parse_character_response().

No LLM, vectorstore, or network required — runs instantly.

Usage:
  pytest tests/test_schema.py -v
"""

import pytest
from rag_pipeline import parse_character_response
from schemas import CharacterResponse


def test_happy_path():
    raw = "SPEECH: I'm so glad you came today.\nMOTION: hug"
    r = parse_character_response(raw)
    assert r.speech == "I'm so glad you came today."
    assert r.motion == "hug"


def test_case_insensitive_tags():
    raw = "speech: Hello Weibo.\nmotion: Wave"
    r = parse_character_response(raw)
    assert r.speech == "Hello Weibo."
    assert r.motion == "wave"


def test_no_speech_tag_uses_full_raw():
    raw = "Hello Weibo, how are you?"
    r = parse_character_response(raw)
    assert r.speech == raw
    assert r.motion == "idle"


def test_no_motion_tag_defaults_idle():
    raw = "SPEECH: Welcome home."
    r = parse_character_response(raw)
    assert r.speech == "Welcome home."
    assert r.motion == "idle"


def test_empty_motion_defaults_idle():
    raw = "SPEECH: Hi!\nMOTION:   "
    r = parse_character_response(raw)
    assert r.motion == "idle"


def test_multi_word_motion_takes_first_token():
    raw = "SPEECH: Come, sit with me.\nMOTION: wave gently"
    r = parse_character_response(raw)
    assert r.motion == "wave"


def test_extra_lines_ignored():
    raw = "Here is my answer:\nSPEECH: I love practicing magic.\nMOTION: smile\nSee you soon!"
    r = parse_character_response(raw)
    assert r.speech == "I love practicing magic."
    assert r.motion == "smile"


def test_model_dump_shape():
    r = parse_character_response("SPEECH: Hi.\nMOTION: nod")
    d = r.model_dump()
    assert set(d.keys()) == {"speech", "motion"}
    assert isinstance(d["speech"], str)
    assert isinstance(d["motion"], str)


def test_returns_character_response_instance():
    r = parse_character_response("SPEECH: Hi.\nMOTION: bow")
    assert isinstance(r, CharacterResponse)
