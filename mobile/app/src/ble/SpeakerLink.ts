/**
 * Transport abstraction between the app and one SoF-inity speaker.
 * BlePlxLink implements it over react-native-ble-plx; MockSpeakerLink
 * simulates speakers for development in a simulator (no Bluetooth).
 */

import {
  clockOffsetUs,
  ClockEstimate,
  decodeFrame,
  encodeSetAlignment,
  encodeGainFrame,
  encodeSyncPing,
  GainFrame,
} from '@sofinity/core';

export interface DiscoveredSpeaker {
  id: string;
  name: string;
  rssi: number | null;
}

export interface SpeakerLink {
  scan(onFound: (speaker: DiscoveredSpeaker) => void, timeoutMs: number): Promise<void>;
  connect(speakerId: string): Promise<void>;
  disconnect(speakerId: string): Promise<void>;
  /** Static per-speaker time alignment + level trim from the acoustic map. */
  setAlignment(speakerId: string, delayUs: number, gainQ15: number): Promise<void>;
  /** One gain keyframe; sent write-without-response for throughput. */
  sendGainFrame(speakerId: string, frame: Omit<GainFrame, 'type'>): Promise<void>;
  /** NTP-style two-way exchange; caller repeats and keeps the lowest RTT. */
  syncClock(speakerId: string): Promise<ClockEstimate>;
}

const nowUs = (): bigint => BigInt(Math.round(Date.now() * 1000));

/**
 * Simulated array for development without hardware: three virtual
 * speakers with fixed clock offsets and a jittery link.
 */
export class MockSpeakerLink implements SpeakerLink {
  private readonly virtualOffsetsUs = new Map<string, bigint>();

  async scan(onFound: (speaker: DiscoveredSpeaker) => void, _timeoutMs: number): Promise<void> {
    const mocks: DiscoveredSpeaker[] = [
      { id: 'mock-0', name: 'SoF-inity L', rssi: -48 },
      { id: 'mock-1', name: 'SoF-inity R', rssi: -52 },
      { id: 'mock-2', name: 'SoF-inity S', rssi: -60 },
    ];
    mocks.forEach((m, i) => {
      this.virtualOffsetsUs.set(m.id, BigInt(1000 * (i + 1)));
      setTimeout(() => onFound(m), 150 * (i + 1));
    });
  }

  async connect(_speakerId: string): Promise<void> {}
  async disconnect(_speakerId: string): Promise<void> {}
  async setAlignment(_speakerId: string, delayUs: number, gainQ15: number): Promise<void> {
    // Exercise the codec so mock runs catch encoding regressions.
    decodeFrame(encodeSetAlignment({ delayUs, gainQ15 }));
  }
  async sendGainFrame(_speakerId: string, frame: Omit<GainFrame, 'type'>): Promise<void> {
    decodeFrame(encodeGainFrame(frame));
  }

  async syncClock(speakerId: string): Promise<ClockEstimate> {
    const offset = this.virtualOffsetsUs.get(speakerId) ?? 0n;
    const jitter = (): bigint => BigInt(500 + Math.round(Math.random() * 300));
    const t1 = nowUs();
    decodeFrame(encodeSyncPing({ t1Us: t1 }));
    const t2 = t1 + jitter() + offset;
    const t3 = t2 + 120n;
    const t4 = t3 - offset + jitter();
    return clockOffsetUs(t1, t2, t3, t4);
  }
}
