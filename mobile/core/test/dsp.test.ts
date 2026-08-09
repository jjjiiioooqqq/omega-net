import { buildAcousticMap, SPEED_OF_SOUND_M_S, SpeakerPlacement, Vec3 } from '../src/geometry';
import { RingPanner, energyVectorErrorRad } from '../src/panner';
import { buildGainSchedules, TrajectoryPoint } from '../src/scheduler';
import { q15ToGain } from '../src/protocol';

const origin: Vec3 = { x: 0, y: 0, z: 0 };

const speaker = (id: number, x: number, y: number, z = 0): SpeakerPlacement => ({
  id: `sp${id}`,
  name: `sp${id}`,
  position: { x, y, z },
});

function ring(count: number, radius: number): SpeakerPlacement[] {
  return Array.from({ length: count }, (_, i) => {
    const a = (2 * Math.PI * i) / count;
    return speaker(i, radius * Math.cos(a), radius * Math.sin(a));
  });
}

describe('buildAcousticMap', () => {
  // Mirrors the Python test: near speaker at 1 m, far at 2 m.
  it('computes delays and gains matching the Python reference', () => {
    const map = buildAcousticMap([speaker(0, 1, 0), speaker(1, 0, 2)], origin);
    expect(map.distancesM[0]).toBeCloseTo(1.0, 10);
    expect(map.distancesM[1]).toBeCloseTo(2.0, 10);
    expect(map.alignmentDelaysS[1]).toBeCloseTo(0.0, 10);
    expect(map.alignmentDelaysS[0]).toBeCloseTo(1.0 / SPEED_OF_SOUND_M_S, 10);
    expect(map.compensationGains[0]).toBeCloseTo(0.5, 10);
    expect(map.compensationGains[1]).toBeCloseTo(1.0, 10);
    expect(map.azimuthsRad[0]).toBeCloseTo(0.0, 10);
    expect(map.azimuthsRad[1]).toBeCloseTo(Math.PI / 2, 10);
  });

  it('rejects degenerate setups', () => {
    expect(() => buildAcousticMap([speaker(0, 1, 0)], origin)).toThrow('at least 2');
    expect(() => buildAcousticMap([speaker(0, 0, 0), speaker(1, 1, 0)], origin)).toThrow(
      'coincides',
    );
  });
});

describe('RingPanner', () => {
  const azimuths = [0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2];

  it('is one-hot at a speaker direction', () => {
    const gains = new RingPanner(azimuths).gains(Math.PI / 2);
    expect(gains[1]).toBeCloseTo(1.0, 10);
    expect(gains[0] + gains[2] + gains[3]).toBeCloseTo(0.0, 10);
  });

  it('is equal-power between an adjacent pair', () => {
    const gains = new RingPanner(azimuths).gains(Math.PI / 4);
    expect(gains[0]).toBeCloseTo(gains[1], 10);
    const power = gains.reduce((s, g) => s + g * g, 0);
    expect(power).toBeCloseTo(1.0, 10);
    expect(energyVectorErrorRad(gains, azimuths, Math.PI / 4)).toBeCloseTo(0.0, 6);
  });

  it('handles the wraparound pair', () => {
    const gains = new RingPanner(azimuths).gains(-Math.PI / 4);
    expect(gains[3]).toBeCloseTo(gains[0], 10);
    expect(gains[1]).toBeCloseTo(0.0, 10);
    expect(gains[2]).toBeCloseTo(0.0, 10);
  });

  it('rejects coincident azimuths', () => {
    expect(() => new RingPanner([0.1, 0.1, 2.0])).toThrow('singular');
  });
});

describe('buildGainSchedules', () => {
  const map = buildAcousticMap(ring(4, 2), origin);

  it('produces a one-hot schedule for a static source at a speaker', () => {
    const trajectory: TrajectoryPoint[] = [
      { timeS: 0, azimuthRad: Math.PI / 2, distanceM: 1 },
      { timeS: 1, azimuthRad: Math.PI / 2, distanceM: 1 },
    ];
    const result = buildGainSchedules(trajectory, map, { updateRateHz: 10 });
    expect(result.schedules).toHaveLength(4);
    expect(result.schedules[0].frames).toHaveLength(11);
    for (const frame of result.schedules[1].frames) {
      expect(q15ToGain(frame.gainQ15)).toBeCloseTo(1.0, 3);
    }
    for (const ch of [0, 2, 3]) {
      for (const frame of result.schedules[ch].frames) {
        expect(frame.gainQ15).toBe(0);
      }
    }
    expect(result.maxEnergyVectorErrorRad).toBeCloseTo(0.0, 6);
  });

  it('sweeps energy between speakers for a moving source', () => {
    const trajectory: TrajectoryPoint[] = [
      { timeS: 0, azimuthRad: 0, distanceM: 1 },
      { timeS: 1, azimuthRad: Math.PI / 2, distanceM: 1 },
    ];
    const { schedules } = buildGainSchedules(trajectory, map, { updateRateHz: 20 });
    const first = schedules.map((s) => s.frames[0].gainQ15);
    const last = schedules.map((s) => s.frames[s.frames.length - 1].gainQ15);
    expect(first[0]).toBeGreaterThan(first[1]);
    expect(last[1]).toBeGreaterThan(last[0]);
    // Timestamps advance at the update rate.
    expect(Number(schedules[0].frames[1].timestampUs)).toBe(50_000);
  });

  it('attenuates with distance but never amplifies', () => {
    const trajectory: TrajectoryPoint[] = [
      { timeS: 0, azimuthRad: 0, distanceM: 0.2 }, // closer than reference
      { timeS: 1, azimuthRad: 0, distanceM: 4 },
    ];
    const { schedules } = buildGainSchedules(trajectory, map, { updateRateHz: 10 });
    const frames = schedules[0].frames;
    expect(q15ToGain(frames[0].gainQ15)).toBeCloseTo(1.0, 3); // clamped at unity
    expect(q15ToGain(frames[frames.length - 1].gainQ15)).toBeCloseTo(0.25, 2);
  });

  it('validates trajectories', () => {
    expect(() => buildGainSchedules([], map)).toThrow('at least one');
    expect(() =>
      buildGainSchedules(
        [
          { timeS: 0, azimuthRad: 0, distanceM: 1 },
          { timeS: 0, azimuthRad: 1, distanceM: 1 },
        ],
        map,
      ),
    ).toThrow('strictly increasing');
    expect(() => buildGainSchedules([{ timeS: 0, azimuthRad: 0, distanceM: 0 }], map)).toThrow(
      'positive',
    );
  });
});
