import React, { useRef, useState } from 'react';
import { Button, ScrollView, StyleSheet, Text, View } from 'react-native';

import { buildGainSchedules, gainToQ15, TrajectoryPoint } from '@sofinity/core';

import { connectedSpeakers, useArrayStore } from '../state/ArrayStore';

const UPDATE_RATE_HZ = 50;
const ORBIT_PERIOD_S = 8;

/** One full orbit around the listener at 2 m, sampled at 4 Hz keypoints. */
function orbitTrajectory(): TrajectoryPoint[] {
  const points: TrajectoryPoint[] = [];
  for (let t = 0; t <= ORBIT_PERIOD_S; t += 0.25) {
    points.push({
      timeS: t,
      azimuthRad: (2 * Math.PI * t) / ORBIT_PERIOD_S,
      distanceM: 2,
    });
  }
  return points;
}

export function PlayScreen() {
  const { link, speakers, acousticMap, acousticMapError } = useArrayStore();
  const connected = connectedSpeakers(speakers);
  const [status, setStatus] = useState<string>('idle');
  const [metric, setMetric] = useState<number | null>(null);
  const stopRequested = useRef(false);

  const applyAlignment = async () => {
    if (!acousticMap) {
      return;
    }
    setStatus('writing alignment…');
    for (let i = 0; i < connected.length; i += 1) {
      await link.setAlignment(
        connected[i].id,
        Math.round(acousticMap.alignmentDelaysS[i] * 1e6),
        gainToQ15(acousticMap.compensationGains[i]),
      );
    }
    setStatus('alignment written');
  };

  const startOrbit = async () => {
    if (!acousticMap) {
      return;
    }
    stopRequested.current = false;
    const result = buildGainSchedules(orbitTrajectory(), acousticMap, {
      updateRateHz: UPDATE_RATE_HZ,
    });
    setMetric(result.maxEnergyVectorErrorRad);
    setStatus('streaming gain schedule…');
    const tickMs = 1000 / UPDATE_RATE_HZ;
    const frameCount = result.schedules[0].frames.length;
    for (let seq = 0; seq < frameCount && !stopRequested.current; seq += 1) {
      const tickStart = Date.now();
      await Promise.all(
        result.schedules.map((schedule, i) =>
          link.sendGainFrame(connected[i].id, schedule.frames[seq]),
        ),
      );
      const elapsed = Date.now() - tickStart;
      if (elapsed < tickMs) {
        await new Promise((r) => setTimeout(r, tickMs - elapsed));
      }
    }
    setStatus(stopRequested.current ? 'stopped' : 'orbit complete');
  };

  const stop = () => {
    stopRequested.current = true;
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.heading}>Playback</Text>
      <Text style={styles.meta}>
        Program audio reaches every speaker as a shared stream (LE Audio broadcast, line-in, or
        Wi-Fi). This app steers the sound: it streams per-speaker gain keyframes at{' '}
        {UPDATE_RATE_HZ} Hz over BLE and each speaker shapes the shared audio locally.
      </Text>
      {acousticMapError && <Text style={styles.error}>{acousticMapError}</Text>}
      <Button title="1. Apply time alignment + level trim" onPress={applyAlignment} disabled={!acousticMap} />
      <Button title="2. Start demo orbit (8 s)" onPress={startOrbit} disabled={!acousticMap} />
      <Button title="Stop" onPress={stop} />
      <View style={styles.statusBox}>
        <Text style={styles.meta}>status: {status}</Text>
        {metric !== null && (
          <Text style={styles.meta}>
            worst-case localization error: {((metric * 180) / Math.PI).toFixed(1)}° (Gerzon energy
            vector)
          </Text>
        )}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: 16, gap: 12 },
  heading: { fontSize: 16, fontWeight: '700' },
  statusBox: { marginTop: 8, gap: 4 },
  meta: { fontSize: 12, color: '#666' },
  error: { color: '#b00020' },
});
