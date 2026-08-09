import json
from pathlib import Path

import pytest

from engint.spatial_audio.ble_protocol import (
    GainFrame,
    SetAlignment,
    SyncPing,
    SyncPong,
    clock_offset_us,
    decode_frame,
    encode_gain_frame,
    encode_set_alignment,
    encode_sync_ping,
    encode_sync_pong,
    gain_to_q15,
    q15_to_gain,
    reference_vectors,
)

FIXTURE = Path(__file__).resolve().parents[1] / "mobile" / "core" / "test" / "fixtures" / "protocol_vectors.json"


def test_q15_round_trip_and_bounds():
    assert gain_to_q15(0.0) == 0
    assert gain_to_q15(1.0) == 32767
    assert q15_to_gain(gain_to_q15(0.5)) == pytest.approx(0.5, abs=1e-4)
    with pytest.raises(ValueError):
        gain_to_q15(1.5)
    with pytest.raises(ValueError):
        q15_to_gain(40000)


def test_frame_round_trips():
    frames = [
        (SetAlignment(delay_us=1458, gain_q15=16384), encode_set_alignment),
        (GainFrame(seq=7, timestamp_us=2**40, gain_q15=100), encode_gain_frame),
        (SyncPing(t1_us=42), encode_sync_ping),
        (SyncPong(t1_us=1, t2_us=2, t3_us=3), encode_sync_pong),
    ]
    for frame, encoder in frames:
        assert decode_frame(encoder(frame)) == frame


def test_decode_rejects_malformed():
    with pytest.raises(ValueError):
        decode_frame(b"")
    with pytest.raises(ValueError):
        decode_frame(bytes([0x7F, 0, 0]))
    truncated = encode_gain_frame(GainFrame(seq=1, timestamp_us=1, gain_q15=1))[:-1]
    with pytest.raises(ValueError):
        decode_frame(truncated)


def test_clock_offset_symmetric_link():
    # Speaker clock 1000 us ahead; 800 us symmetric one-way latency.
    t1, one_way, offset_true = 5_000_000, 800, 1000
    t2 = t1 + one_way + offset_true
    t3 = t2 + 150  # processing time on the speaker
    t4 = t3 - offset_true + one_way
    offset, rtt = clock_offset_us(t1, t2, t3, t4)
    assert offset == pytest.approx(offset_true)
    assert rtt == pytest.approx(2 * one_way)


def test_committed_fixture_matches_reference_vectors():
    """The JSON fixture shared with the TypeScript codec must not drift."""
    assert FIXTURE.exists(), "run tests/generate fixture: see reference_vectors()"
    committed = json.loads(FIXTURE.read_text())
    assert committed == reference_vectors()
