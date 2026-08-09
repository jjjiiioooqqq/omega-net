import React, { useState } from 'react';
import { Button, FlatList, StyleSheet, Switch, Text, View } from 'react-native';

import { DiscoveredSpeaker } from '../ble/SpeakerLink';
import { useArrayStore } from '../state/ArrayStore';

const SCAN_TIMEOUT_MS = 8000;

export function DiscoverScreen() {
  const { link, useMockLink, setUseMockLink, speakers, setSpeakers } = useArrayStore();
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onFound = (found: DiscoveredSpeaker) => {
    setSpeakers((prev) =>
      prev.some((s) => s.id === found.id)
        ? prev
        : [
            ...prev,
            {
              id: found.id,
              name: found.name,
              position: { x: 0, y: 0, z: 0 },
              connected: false,
            },
          ],
    );
  };

  const scan = async () => {
    setScanning(true);
    setError(null);
    try {
      await link.scan(onFound, SCAN_TIMEOUT_MS);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setScanning(false);
    }
  };

  const toggleConnection = async (id: string, connected: boolean) => {
    setError(null);
    try {
      if (connected) {
        await link.disconnect(id);
      } else {
        await link.connect(id);
        const clock = await link.syncClock(id);
        setSpeakers((prev) => prev.map((s) => (s.id === id ? { ...s, clock } : s)));
      }
      setSpeakers((prev) => prev.map((s) => (s.id === id ? { ...s, connected: !connected } : s)));
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.row}>
        <Text style={styles.label}>Simulated speakers (no Bluetooth)</Text>
        <Switch value={useMockLink} onValueChange={setUseMockLink} />
      </View>
      <Button title={scanning ? 'Scanning…' : 'Scan for SoF-inity speakers'} onPress={scan} disabled={scanning} />
      {error && <Text style={styles.error}>{error}</Text>}
      <FlatList
        data={speakers}
        keyExtractor={(s) => s.id}
        renderItem={({ item }) => (
          <View style={styles.speakerRow}>
            <View style={styles.speakerInfo}>
              <Text style={styles.speakerName}>{item.name}</Text>
              {item.clock && (
                <Text style={styles.meta}>
                  clock offset {Math.round(item.clock.offsetUs)} µs · rtt{' '}
                  {Math.round(item.clock.roundTripUs)} µs
                </Text>
              )}
            </View>
            <Button
              title={item.connected ? 'Disconnect' : 'Connect'}
              onPress={() => toggleConnection(item.id, item.connected)}
            />
          </View>
        )}
        ListEmptyComponent={<Text style={styles.meta}>No speakers found yet.</Text>}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 16, gap: 12 },
  row: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  label: { fontSize: 14 },
  error: { color: '#b00020' },
  speakerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
  },
  speakerInfo: { flex: 1 },
  speakerName: { fontSize: 16, fontWeight: '600' },
  meta: { fontSize: 12, color: '#666' },
});
