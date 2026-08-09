"""SoF-inity BLE GATT control protocol (speaker-firmware reference codec).

Transport model (explicit, because Bluetooth constrains the design):
classic A2DP is stereo and point-to-point, so a phone cannot stream N
independent audio channels to N speakers. SoF-inity therefore streams
*parameters*, not multichannel audio: the app computes the acoustic map
and per-speaker gain schedules and sends them over BLE GATT (a few
kB/s for an 8-speaker array at 50 Hz updates), while program audio
reaches every speaker as a shared stream (LE Audio broadcast / line-in
/ Wi-Fi). Each speaker applies its own alignment delay and gain
envelope to the shared audio.

All frames are little-endian. Gains are Q15 (0..32767 == 0.0..1.0).
Timestamps are microseconds on the sender's monotonic clock; the sync
handshake (NTP-style two-way exchange) lets each speaker convert phone
timestamps into its own clock.

Frame layout (type byte first):
  0x01 SET_ALIGNMENT  u32 delay_us, u16 gain_q15               (7 bytes)
  0x02 GAIN_FRAME     u16 seq, u64 timestamp_us, u16 gain_q15  (13 bytes)
  0x03 SYNC_PING      u64 t1_us                                (9 bytes)
  0x04 SYNC_PONG      u64 t1_us, u64 t2_us, u64 t3_us          (25 bytes)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

PROTOCOL_VERSION = 1

SERVICE_UUID = "50f1a5d0-0001-4e0b-bf3a-6f1c2d3e4a01"
CHAR_DEVICE_INFO_UUID = "50f1a5d0-0002-4e0b-bf3a-6f1c2d3e4a01"
CHAR_ALIGNMENT_UUID = "50f1a5d0-0003-4e0b-bf3a-6f1c2d3e4a01"
CHAR_GAIN_STREAM_UUID = "50f1a5d0-0004-4e0b-bf3a-6f1c2d3e4a01"
CHAR_SYNC_UUID = "50f1a5d0-0005-4e0b-bf3a-6f1c2d3e4a01"

FRAME_SET_ALIGNMENT = 0x01
FRAME_GAIN = 0x02
FRAME_SYNC_PING = 0x03
FRAME_SYNC_PONG = 0x04

Q15_ONE = 32767


def gain_to_q15(gain: float) -> int:
    if not (0.0 <= gain <= 1.0):
        raise ValueError("gain must be in [0, 1]")
    return round(gain * Q15_ONE)


def q15_to_gain(q15: int) -> float:
    if not (0 <= q15 <= Q15_ONE):
        raise ValueError("q15 value out of range")
    return q15 / Q15_ONE


@dataclass(slots=True)
class SetAlignment:
    delay_us: int
    gain_q15: int


@dataclass(slots=True)
class GainFrame:
    seq: int
    timestamp_us: int
    gain_q15: int


@dataclass(slots=True)
class SyncPing:
    t1_us: int


@dataclass(slots=True)
class SyncPong:
    t1_us: int
    t2_us: int
    t3_us: int


def encode_set_alignment(frame: SetAlignment) -> bytes:
    return struct.pack("<BIH", FRAME_SET_ALIGNMENT, frame.delay_us, frame.gain_q15)


def encode_gain_frame(frame: GainFrame) -> bytes:
    return struct.pack("<BHQH", FRAME_GAIN, frame.seq, frame.timestamp_us, frame.gain_q15)


def encode_sync_ping(frame: SyncPing) -> bytes:
    return struct.pack("<BQ", FRAME_SYNC_PING, frame.t1_us)


def encode_sync_pong(frame: SyncPong) -> bytes:
    return struct.pack("<BQQQ", FRAME_SYNC_PONG, frame.t1_us, frame.t2_us, frame.t3_us)


def decode_frame(data: bytes) -> SetAlignment | GainFrame | SyncPing | SyncPong:
    if not data:
        raise ValueError("empty frame")
    frame_type = data[0]
    if frame_type == FRAME_SET_ALIGNMENT:
        delay_us, gain_q15 = struct.unpack("<IH", _payload(data, 6))
        return SetAlignment(delay_us=delay_us, gain_q15=gain_q15)
    if frame_type == FRAME_GAIN:
        seq, timestamp_us, gain_q15 = struct.unpack("<HQH", _payload(data, 12))
        return GainFrame(seq=seq, timestamp_us=timestamp_us, gain_q15=gain_q15)
    if frame_type == FRAME_SYNC_PING:
        (t1_us,) = struct.unpack("<Q", _payload(data, 8))
        return SyncPing(t1_us=t1_us)
    if frame_type == FRAME_SYNC_PONG:
        t1_us, t2_us, t3_us = struct.unpack("<QQQ", _payload(data, 24))
        return SyncPong(t1_us=t1_us, t2_us=t2_us, t3_us=t3_us)
    raise ValueError(f"unknown frame type 0x{frame_type:02x}")


def _payload(data: bytes, expected_len: int) -> bytes:
    payload = data[1:]
    if len(payload) != expected_len:
        raise ValueError(f"frame payload length {len(payload)}, expected {expected_len}")
    return payload


def clock_offset_us(t1_us: int, t2_us: int, t3_us: int, t4_us: int) -> tuple[float, float]:
    """NTP-style offset estimate from a two-way exchange.

    t1: phone send, t2: speaker receive, t3: speaker reply, t4: phone
    receive. Returns (offset_us, round_trip_us) where speaker_clock ~=
    phone_clock + offset. Accuracy is bounded by link asymmetry, which
    BLE does not let us measure — report round-trip so callers can
    reject noisy exchanges.
    """
    offset = ((t2_us - t1_us) + (t3_us - t4_us)) / 2.0
    round_trip = (t4_us - t1_us) - (t3_us - t2_us)
    if round_trip < 0:
        raise ValueError("negative round trip; timestamps are inconsistent")
    return offset, float(round_trip)


def reference_vectors() -> list[dict]:
    """Golden encode vectors shared with the TypeScript codec.

    The committed fixture mobile/core/test/fixtures/protocol_vectors.json
    must equal this list; both codecs test against it so they cannot
    drift apart silently.
    """
    frames: list[tuple[str, SetAlignment | GainFrame | SyncPing | SyncPong, bytes]] = [
        (
            "set_alignment_typical",
            SetAlignment(delay_us=1458, gain_q15=gain_to_q15(0.5)),
            encode_set_alignment(SetAlignment(delay_us=1458, gain_q15=gain_to_q15(0.5))),
        ),
        (
            "set_alignment_extremes",
            SetAlignment(delay_us=0xFFFFFFFF, gain_q15=Q15_ONE),
            encode_set_alignment(SetAlignment(delay_us=0xFFFFFFFF, gain_q15=Q15_ONE)),
        ),
        (
            "gain_frame_typical",
            GainFrame(seq=513, timestamp_us=1_699_999_999_123_456, gain_q15=gain_to_q15(0.25)),
            encode_gain_frame(
                GainFrame(seq=513, timestamp_us=1_699_999_999_123_456, gain_q15=gain_to_q15(0.25))
            ),
        ),
        (
            "gain_frame_zero",
            GainFrame(seq=0, timestamp_us=0, gain_q15=0),
            encode_gain_frame(GainFrame(seq=0, timestamp_us=0, gain_q15=0)),
        ),
        (
            "sync_ping",
            SyncPing(t1_us=123_456_789_012),
            encode_sync_ping(SyncPing(t1_us=123_456_789_012)),
        ),
        (
            "sync_pong",
            SyncPong(t1_us=123_456_789_012, t2_us=123_456_790_500, t3_us=123_456_791_000),
            encode_sync_pong(
                SyncPong(t1_us=123_456_789_012, t2_us=123_456_790_500, t3_us=123_456_791_000)
            ),
        ),
    ]
    vectors = []
    for name, frame, encoded in frames:
        fields = {slot: int(getattr(frame, slot)) for slot in frame.__dataclass_fields__}
        vectors.append(
            {
                "name": name,
                "frame_type": type(frame).__name__,
                "fields": fields,
                "hex": encoded.hex(),
            }
        )
    return vectors
