/**
 * SpeakerLink over react-native-ble-plx (real Bluetooth LE).
 *
 * GATT layout is defined by @sofinity/core protocol constants: one
 * primary service, alignment + gain-stream + sync characteristics.
 * Gain keyframes go write-without-response (fits well under the
 * 23-byte minimum ATT MTU; ~13 bytes each, 50 Hz per speaker is a few
 * hundred bytes/s per link — comfortably within BLE bandwidth).
 */

import { BleManager, Device } from 'react-native-ble-plx';
import { Buffer } from 'buffer';

import {
  CHAR_ALIGNMENT_UUID,
  CHAR_GAIN_STREAM_UUID,
  CHAR_SYNC_UUID,
  SERVICE_UUID,
  clockOffsetUs,
  ClockEstimate,
  decodeFrame,
  encodeGainFrame,
  encodeSetAlignment,
  encodeSyncPing,
  GainFrame,
} from '@sofinity/core';

import { DiscoveredSpeaker, SpeakerLink } from './SpeakerLink';

const toBase64 = (bytes: Uint8Array): string => Buffer.from(bytes).toString('base64');
const fromBase64 = (b64: string): Uint8Array => new Uint8Array(Buffer.from(b64, 'base64'));
const nowUs = (): bigint => BigInt(Math.round(Date.now() * 1000));

export class BlePlxLink implements SpeakerLink {
  private readonly manager = new BleManager();
  private readonly devices = new Map<string, Device>();

  async scan(onFound: (speaker: DiscoveredSpeaker) => void, timeoutMs: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const seen = new Set<string>();
      this.manager.startDeviceScan([SERVICE_UUID], null, (error, device) => {
        if (error) {
          this.manager.stopDeviceScan();
          reject(error);
          return;
        }
        if (device && !seen.has(device.id)) {
          seen.add(device.id);
          onFound({ id: device.id, name: device.name ?? device.id, rssi: device.rssi });
        }
      });
      setTimeout(() => {
        this.manager.stopDeviceScan();
        resolve();
      }, timeoutMs);
    });
  }

  async connect(speakerId: string): Promise<void> {
    const device = await this.manager.connectToDevice(speakerId);
    await device.discoverAllServicesAndCharacteristics();
    this.devices.set(speakerId, device);
  }

  async disconnect(speakerId: string): Promise<void> {
    await this.devices.get(speakerId)?.cancelConnection();
    this.devices.delete(speakerId);
  }

  async setAlignment(speakerId: string, delayUs: number, gainQ15: number): Promise<void> {
    await this.device(speakerId).writeCharacteristicWithResponseForService(
      SERVICE_UUID,
      CHAR_ALIGNMENT_UUID,
      toBase64(encodeSetAlignment({ delayUs, gainQ15 })),
    );
  }

  async sendGainFrame(speakerId: string, frame: Omit<GainFrame, 'type'>): Promise<void> {
    await this.device(speakerId).writeCharacteristicWithoutResponseForService(
      SERVICE_UUID,
      CHAR_GAIN_STREAM_UUID,
      toBase64(encodeGainFrame(frame)),
    );
  }

  async syncClock(speakerId: string): Promise<ClockEstimate> {
    const device = this.device(speakerId);
    const t1 = nowUs();
    await device.writeCharacteristicWithResponseForService(
      SERVICE_UUID,
      CHAR_SYNC_UUID,
      toBase64(encodeSyncPing({ t1Us: t1 })),
    );
    const readBack = await device.readCharacteristicForService(SERVICE_UUID, CHAR_SYNC_UUID);
    const t4 = nowUs();
    const pong = decodeFrame(fromBase64(readBack.value ?? ''));
    if (pong.type !== 'SyncPong' || pong.t1Us !== t1) {
      throw new Error('speaker returned an invalid or stale sync pong');
    }
    return clockOffsetUs(pong.t1Us, pong.t2Us, pong.t3Us, t4);
  }

  private device(speakerId: string): Device {
    const device = this.devices.get(speakerId);
    if (!device) {
      throw new Error(`speaker ${speakerId} is not connected`);
    }
    return device;
  }
}
