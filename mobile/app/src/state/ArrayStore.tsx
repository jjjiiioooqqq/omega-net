/**
 * App-wide state: the speaker array being configured, the listening
 * position, and derived acoustic data. The acoustic map is recomputed
 * from positions on every read — it is derived truth, never cached
 * state that could go stale.
 */

import React, { createContext, useContext, useMemo, useState } from 'react';

import { AcousticMap, buildAcousticMap, ClockEstimate, Vec3 } from '@sofinity/core';

import { BlePlxLink } from '../ble/BlePlxLink';
import { MockSpeakerLink, SpeakerLink } from '../ble/SpeakerLink';

export interface ArraySpeaker {
  id: string;
  name: string;
  position: Vec3;
  connected: boolean;
  clock?: ClockEstimate;
}

interface ArrayState {
  link: SpeakerLink;
  useMockLink: boolean;
  setUseMockLink: (mock: boolean) => void;
  speakers: ArraySpeaker[];
  setSpeakers: React.Dispatch<React.SetStateAction<ArraySpeaker[]>>;
  listener: Vec3;
  setListener: (v: Vec3) => void;
  /** null until the array is valid (>= 2 speakers, none on the listener). */
  acousticMap: AcousticMap | null;
  acousticMapError: string | null;
}

const ArrayContext = createContext<ArrayState | null>(null);

export function ArrayStoreProvider({ children }: { children: React.ReactNode }) {
  const [useMockLink, setUseMockLink] = useState(true);
  const [speakers, setSpeakers] = useState<ArraySpeaker[]>([]);
  const [listener, setListener] = useState<Vec3>({ x: 0, y: 0, z: 0 });

  const link = useMemo<SpeakerLink>(
    () => (useMockLink ? new MockSpeakerLink() : new BlePlxLink()),
    [useMockLink],
  );

  const { acousticMap, acousticMapError } = useMemo(() => {
    const connected = speakers.filter((s) => s.connected);
    if (connected.length < 2) {
      return { acousticMap: null, acousticMapError: 'connect at least 2 speakers' };
    }
    try {
      return {
        acousticMap: buildAcousticMap(
          connected.map((s) => ({ id: s.id, name: s.name, position: s.position })),
          listener,
        ),
        acousticMapError: null,
      };
    } catch (e) {
      return { acousticMap: null, acousticMapError: (e as Error).message };
    }
  }, [speakers, listener]);

  const value: ArrayState = {
    link,
    useMockLink,
    setUseMockLink,
    speakers,
    setSpeakers,
    listener,
    setListener,
    acousticMap,
    acousticMapError,
  };
  return <ArrayContext.Provider value={value}>{children}</ArrayContext.Provider>;
}

export function useArrayStore(): ArrayState {
  const state = useContext(ArrayContext);
  if (!state) {
    throw new Error('useArrayStore must be used inside ArrayStoreProvider');
  }
  return state;
}

/** Connected speakers in acoustic-map index order. */
export function connectedSpeakers(speakers: ArraySpeaker[]): ArraySpeaker[] {
  return speakers.filter((s) => s.connected);
}
