import React from 'react';
import { ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';

import { Vec3 } from '@sofinity/core';

import { connectedSpeakers, useArrayStore } from '../state/ArrayStore';

function AxisInput({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <View style={styles.axis}>
      <Text style={styles.axisLabel}>{label}</Text>
      <TextInput
        style={styles.input}
        keyboardType="numbers-and-punctuation"
        defaultValue={String(value)}
        onEndEditing={(e) => {
          const parsed = Number(e.nativeEvent.text);
          if (Number.isFinite(parsed)) {
            onChange(parsed);
          }
        }}
      />
    </View>
  );
}

function PositionEditor({
  name,
  position,
  onChange,
}: {
  name: string;
  position: Vec3;
  onChange: (v: Vec3) => void;
}) {
  return (
    <View style={styles.editorRow}>
      <Text style={styles.name}>{name}</Text>
      <AxisInput label="x" value={position.x} onChange={(x) => onChange({ ...position, x })} />
      <AxisInput label="y" value={position.y} onChange={(y) => onChange({ ...position, y })} />
      <AxisInput label="z" value={position.z} onChange={(z) => onChange({ ...position, z })} />
    </View>
  );
}

export function LayoutScreen() {
  const { speakers, setSpeakers, listener, setListener, acousticMap, acousticMapError } =
    useArrayStore();
  const connected = connectedSpeakers(speakers);

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.heading}>Measured positions (meters, room frame)</Text>
      <Text style={styles.meta}>
        Enter each speaker's measured position. The acoustic map below is derived from these
        measurements — delays and trims update as you type.
      </Text>
      <PositionEditor name="Listener" position={listener} onChange={setListener} />
      {connected.map((s) => (
        <PositionEditor
          key={s.id}
          name={s.name}
          position={s.position}
          onChange={(position) =>
            setSpeakers((prev) => prev.map((p) => (p.id === s.id ? { ...p, position } : p)))
          }
        />
      ))}
      {connected.length === 0 && <Text style={styles.meta}>Connect speakers first.</Text>}

      <Text style={styles.heading}>Acoustic map</Text>
      {acousticMapError && <Text style={styles.error}>{acousticMapError}</Text>}
      {acousticMap &&
        connected.map((s, i) => (
          <View key={s.id} style={styles.mapRow}>
            <Text style={styles.name}>{s.name}</Text>
            <Text style={styles.meta}>
              {acousticMap.distancesM[i].toFixed(2)} m · align{' '}
              {(acousticMap.alignmentDelaysS[i] * 1e6).toFixed(0)} µs · trim{' '}
              {(20 * Math.log10(acousticMap.compensationGains[i])).toFixed(1)} dB · az{' '}
              {((acousticMap.azimuthsRad[i] * 180) / Math.PI).toFixed(0)}°
            </Text>
          </View>
        ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: 16, gap: 10 },
  heading: { fontSize: 16, fontWeight: '700', marginTop: 12 },
  editorRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  name: { width: 90, fontSize: 14, fontWeight: '600' },
  axis: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  axisLabel: { fontSize: 12, color: '#666' },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 4,
    paddingHorizontal: 6,
    paddingVertical: 2,
    minWidth: 52,
  },
  mapRow: { paddingVertical: 4 },
  meta: { fontSize: 12, color: '#666' },
  error: { color: '#b00020' },
});
