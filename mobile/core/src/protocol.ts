/**
 * SoF-inity BLE GATT control protocol codec (app side).
 *
 * Mirrors engint.spatial_audio.ble_protocol (Python, the speaker
 * firmware reference). Both codecs are tested against the same
 * committed golden vectors (test/fixtures/protocol_vectors.json) so
 * they cannot drift apart silently.
 *
 * All frames little-endian; gains Q15; timestamps microseconds.
 */

export const PROTOCOL_VERSION = 1;

export const SERVICE_UUID = '50f1a5d0-0001-4e0b-bf3a-6f1c2d3e4a01';
export const CHAR_DEVICE_INFO_UUID = '50f1a5d0-0002-4e0b-bf3a-6f1c2d3e4a01';
export const CHAR_ALIGNMENT_UUID = '50f1a5d0-0003-4e0b-bf3a-6f1c2d3e4a01';
export const CHAR_GAIN_STREAM_UUID = '50f1a5d0-0004-4e0b-bf3a-6f1c2d3e4a01';
export const CHAR_SYNC_UUID = '50f1a5d0-0005-4e0b-bf3a-6f1c2d3e4a01';

export const FRAME_SET_ALIGNMENT = 0x01;
export const FRAME_GAIN = 0x02;
export const FRAME_SYNC_PING = 0x03;
export const FRAME_SYNC_PONG = 0x04;

export const Q15_ONE = 32767;

export interface SetAlignment {
  type: 'SetAlignment';
  delayUs: number;
  gainQ15: number;
}

export interface GainFrame {
  type: 'GainFrame';
  seq: number;
  timestampUs: bigint;
  gainQ15: number;
}

export interface SyncPing {
  type: 'SyncPing';
  t1Us: bigint;
}

export interface SyncPong {
  type: 'SyncPong';
  t1Us: bigint;
  t2Us: bigint;
  t3Us: bigint;
}

export type Frame = SetAlignment | GainFrame | SyncPing | SyncPong;

export function gainToQ15(gain: number): number {
  if (!(gain >= 0 && gain <= 1)) {
    throw new Error('gain must be in [0, 1]');
  }
  return Math.round(gain * Q15_ONE);
}

export function q15ToGain(q15: number): number {
  if (!Number.isInteger(q15) || q15 < 0 || q15 > Q15_ONE) {
    throw new Error('q15 value out of range');
  }
  return q15 / Q15_ONE;
}

export function encodeSetAlignment(frame: Omit<SetAlignment, 'type'>): Uint8Array {
  const view = new DataView(new ArrayBuffer(7));
  view.setUint8(0, FRAME_SET_ALIGNMENT);
  view.setUint32(1, frame.delayUs, true);
  view.setUint16(5, frame.gainQ15, true);
  return new Uint8Array(view.buffer);
}

export function encodeGainFrame(frame: Omit<GainFrame, 'type'>): Uint8Array {
  const view = new DataView(new ArrayBuffer(13));
  view.setUint8(0, FRAME_GAIN);
  view.setUint16(1, frame.seq, true);
  view.setBigUint64(3, frame.timestampUs, true);
  view.setUint16(11, frame.gainQ15, true);
  return new Uint8Array(view.buffer);
}

export function encodeSyncPing(frame: Omit<SyncPing, 'type'>): Uint8Array {
  const view = new DataView(new ArrayBuffer(9));
  view.setUint8(0, FRAME_SYNC_PING);
  view.setBigUint64(1, frame.t1Us, true);
  return new Uint8Array(view.buffer);
}

export function encodeSyncPong(frame: Omit<SyncPong, 'type'>): Uint8Array {
  const view = new DataView(new ArrayBuffer(25));
  view.setUint8(0, FRAME_SYNC_PONG);
  view.setBigUint64(1, frame.t1Us, true);
  view.setBigUint64(9, frame.t2Us, true);
  view.setBigUint64(17, frame.t3Us, true);
  return new Uint8Array(view.buffer);
}

export function decodeFrame(data: Uint8Array): Frame {
  if (data.length === 0) {
    throw new Error('empty frame');
  }
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const frameType = view.getUint8(0);
  const expect = (payloadLen: number) => {
    if (data.length !== payloadLen + 1) {
      throw new Error(`frame payload length ${data.length - 1}, expected ${payloadLen}`);
    }
  };
  switch (frameType) {
    case FRAME_SET_ALIGNMENT:
      expect(6);
      return {
        type: 'SetAlignment',
        delayUs: view.getUint32(1, true),
        gainQ15: view.getUint16(5, true),
      };
    case FRAME_GAIN:
      expect(12);
      return {
        type: 'GainFrame',
        seq: view.getUint16(1, true),
        timestampUs: view.getBigUint64(3, true),
        gainQ15: view.getUint16(11, true),
      };
    case FRAME_SYNC_PING:
      expect(8);
      return { type: 'SyncPing', t1Us: view.getBigUint64(1, true) };
    case FRAME_SYNC_PONG:
      expect(24);
      return {
        type: 'SyncPong',
        t1Us: view.getBigUint64(1, true),
        t2Us: view.getBigUint64(9, true),
        t3Us: view.getBigUint64(17, true),
      };
    default:
      throw new Error(`unknown frame type 0x${frameType.toString(16)}`);
  }
}

export interface ClockEstimate {
  /** speakerClock ~= phoneClock + offsetUs */
  offsetUs: number;
  roundTripUs: number;
}

/**
 * NTP-style offset from a two-way exchange (t1 phone send, t2 speaker
 * receive, t3 speaker reply, t4 phone receive). Accuracy is bounded by
 * link asymmetry, which BLE cannot measure — callers should reject
 * exchanges with large round trips.
 */
export function clockOffsetUs(t1: bigint, t2: bigint, t3: bigint, t4: bigint): ClockEstimate {
  const offsetUs = Number((t2 - t1) + (t3 - t4)) / 2;
  const roundTripUs = Number((t4 - t1) - (t3 - t2));
  if (roundTripUs < 0) {
    throw new Error('negative round trip; timestamps are inconsistent');
  }
  return { offsetUs, roundTripUs };
}
