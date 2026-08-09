/**
 * Acoustic mapping: measured speaker geometry relative to one listening
 * position. Mirrors engint.spatial_audio.acoustic_map (Python) — keep
 * the two in sync.
 */

export const SPEED_OF_SOUND_M_S = 343.0;

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface SpeakerPlacement {
  id: string;
  name: string;
  position: Vec3;
}

export interface AcousticMap {
  /** Straight-line speaker->listener distance, meters. */
  distancesM: number[];
  /** Acoustic travel time speaker->listener, seconds. */
  propagationDelaysS: number[];
  /**
   * Extra electronic delay per speaker so every wavefront arrives at the
   * listener simultaneously (closer speakers delayed more, farthest 0).
   */
  alignmentDelaysS: number[];
  /** Level compensation (1/r referenced), normalized so max is 1.0. */
  compensationGains: number[];
  /** Horizontal-plane angle listener->speaker, atan2(y, x), radians. */
  azimuthsRad: number[];
}

export function buildAcousticMap(
  speakers: SpeakerPlacement[],
  listener: Vec3,
  speedOfSoundMS: number = SPEED_OF_SOUND_M_S,
  minDistanceM = 1e-3,
): AcousticMap {
  if (speakers.length < 2) {
    throw new Error('layout requires at least 2 speakers');
  }
  if (speedOfSoundMS <= 0) {
    throw new Error('speed of sound must be positive');
  }
  const distancesM: number[] = [];
  const azimuthsRad: number[] = [];
  for (const s of speakers) {
    const dx = s.position.x - listener.x;
    const dy = s.position.y - listener.y;
    const dz = s.position.z - listener.z;
    const d = Math.hypot(dx, dy, dz);
    if (d < minDistanceM) {
      throw new Error(`speaker '${s.name}' coincides with the listening position`);
    }
    distancesM.push(d);
    azimuthsRad.push(Math.atan2(dy, dx));
  }
  const propagationDelaysS = distancesM.map((d) => d / speedOfSoundMS);
  const maxDelay = Math.max(...propagationDelaysS);
  const maxDistance = Math.max(...distancesM);
  return {
    distancesM,
    propagationDelaysS,
    alignmentDelaysS: propagationDelaysS.map((d) => maxDelay - d),
    compensationGains: distancesM.map((d) => d / maxDistance),
    azimuthsRad,
  };
}
