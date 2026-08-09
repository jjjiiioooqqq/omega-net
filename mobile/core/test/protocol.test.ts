import * as fs from 'fs';
import * as path from 'path';

import {
  clockOffsetUs,
  decodeFrame,
  encodeGainFrame,
  encodeSetAlignment,
  encodeSyncPing,
  encodeSyncPong,
  Frame,
  gainToQ15,
  q15ToGain,
} from '../src/protocol';

interface Vector {
  name: string;
  frame_type: string;
  fields: Record<string, number>;
  hex: string;
}

const vectors: Vector[] = JSON.parse(
  fs.readFileSync(path.join(__dirname, 'fixtures', 'protocol_vectors.json'), 'utf8'),
);

const toHex = (bytes: Uint8Array): string =>
  Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');

function encodeFromVector(v: Vector): Uint8Array {
  switch (v.frame_type) {
    case 'SetAlignment':
      return encodeSetAlignment({ delayUs: v.fields.delay_us, gainQ15: v.fields.gain_q15 });
    case 'GainFrame':
      return encodeGainFrame({
        seq: v.fields.seq,
        timestampUs: BigInt(v.fields.timestamp_us),
        gainQ15: v.fields.gain_q15,
      });
    case 'SyncPing':
      return encodeSyncPing({ t1Us: BigInt(v.fields.t1_us) });
    case 'SyncPong':
      return encodeSyncPong({
        t1Us: BigInt(v.fields.t1_us),
        t2Us: BigInt(v.fields.t2_us),
        t3Us: BigInt(v.fields.t3_us),
      });
    default:
      throw new Error(`unknown vector frame type ${v.frame_type}`);
  }
}

const fromHex = (hex: string): Uint8Array =>
  new Uint8Array(hex.match(/.{2}/g)!.map((b) => parseInt(b, 16)));

describe('golden vectors shared with the Python codec', () => {
  it('has vectors for every frame type', () => {
    const types = new Set(vectors.map((v) => v.frame_type));
    expect(types).toEqual(new Set(['SetAlignment', 'GainFrame', 'SyncPing', 'SyncPong']));
  });

  it.each(vectors.map((v) => [v.name, v] as const))('encodes %s identically', (_name, v) => {
    expect(toHex(encodeFromVector(v))).toBe(v.hex);
  });

  it.each(vectors.map((v) => [v.name, v] as const))('decodes %s identically', (_name, v) => {
    const frame: Frame = decodeFrame(fromHex(v.hex));
    expect(frame.type).toBe(v.frame_type);
    for (const [snakeKey, value] of Object.entries(v.fields)) {
      const camelKey = snakeKey.replace(/_([a-z0-9])/g, (_, c) => c.toUpperCase());
      const decoded = (frame as unknown as Record<string, number | bigint>)[camelKey];
      expect(typeof decoded === 'bigint' ? decoded : BigInt(decoded)).toBe(BigInt(value));
    }
  });
});

describe('codec behavior', () => {
  it('round-trips q15 gains', () => {
    expect(gainToQ15(0)).toBe(0);
    expect(gainToQ15(1)).toBe(32767);
    expect(q15ToGain(gainToQ15(0.5))).toBeCloseTo(0.5, 4);
    expect(() => gainToQ15(1.5)).toThrow();
    expect(() => q15ToGain(40000)).toThrow();
  });

  it('rejects malformed frames', () => {
    expect(() => decodeFrame(new Uint8Array([]))).toThrow('empty');
    expect(() => decodeFrame(new Uint8Array([0x7f, 0, 0]))).toThrow('unknown frame type');
    const truncated = encodeGainFrame({ seq: 1, timestampUs: 1n, gainQ15: 1 }).slice(0, -1);
    expect(() => decodeFrame(truncated)).toThrow('payload length');
  });

  it('estimates clock offset over a symmetric link', () => {
    // Speaker clock 1000 us ahead, 800 us one-way latency, 150 us processing.
    const t1 = 5_000_000n;
    const t2 = t1 + 800n + 1000n;
    const t3 = t2 + 150n;
    const t4 = t3 - 1000n + 800n;
    const { offsetUs, roundTripUs } = clockOffsetUs(t1, t2, t3, t4);
    expect(offsetUs).toBeCloseTo(1000, 6);
    expect(roundTripUs).toBeCloseTo(1600, 6);
  });
});
