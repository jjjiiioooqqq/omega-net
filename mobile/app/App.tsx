import React, { useState } from 'react';
import { Pressable, SafeAreaView, StatusBar, StyleSheet, Text, View } from 'react-native';

import { ArrayStoreProvider } from './src/state/ArrayStore';
import { DiscoverScreen } from './src/screens/DiscoverScreen';
import { LayoutScreen } from './src/screens/LayoutScreen';
import { PlayScreen } from './src/screens/PlayScreen';

const TABS = ['Speakers', 'Layout', 'Play'] as const;
type Tab = (typeof TABS)[number];

export default function App() {
  const [tab, setTab] = useState<Tab>('Speakers');
  return (
    <ArrayStoreProvider>
      <SafeAreaView style={styles.root}>
        <StatusBar barStyle="dark-content" />
        <Text style={styles.title}>SoF-inity</Text>
        <View style={styles.tabs}>
          {TABS.map((t) => (
            <Pressable key={t} style={[styles.tab, tab === t && styles.tabActive]} onPress={() => setTab(t)}>
              <Text style={tab === t ? styles.tabTextActive : styles.tabText}>{t}</Text>
            </Pressable>
          ))}
        </View>
        {tab === 'Speakers' && <DiscoverScreen />}
        {tab === 'Layout' && <LayoutScreen />}
        {tab === 'Play' && <PlayScreen />}
      </SafeAreaView>
    </ArrayStoreProvider>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#fff' },
  title: { fontSize: 20, fontWeight: '800', textAlign: 'center', paddingVertical: 8 },
  tabs: { flexDirection: 'row', borderBottomWidth: 1, borderColor: '#ddd' },
  tab: { flex: 1, paddingVertical: 10, alignItems: 'center' },
  tabActive: { borderBottomWidth: 2, borderColor: '#1a73e8' },
  tabText: { color: '#666' },
  tabTextActive: { color: '#1a73e8', fontWeight: '700' },
});
