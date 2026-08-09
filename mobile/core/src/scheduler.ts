/**
 * Spatiotemporal multiplexing over BLE: instead of streaming N audio
 * channels (impossible over classic Bluetooth), the app samples the
 * source trajectory at the update rate and emits one Q15 gain keyframe
 * per speaker per tick. Speakers linearly interpolate between keyframes
 * — the same block-boundary-interpolation scheme as the offline
 * SpatiotemporalMultiplexer, with the block size set by updateRateHz.
 */

import { AcousticMap } from './geometry';
import { RingPanner, energyVectorErrorRad } from './panner';
import { GainFrame, gainToQ15 } from './protocol';

export interface TrajectoryPoint {
  timeS: number;
  azimuthRad: number;
  /** Source distance from the listener, meters (drives 1/r level). */
  distanceM: number;
}

export interface SpeakerSchedule {
  speakerIndex: number;
  frames: Omit<GainFrame, 'type'>[];
}

export interface ScheduleResult {
  schedules: SpeakerSchedule[];
  /** Worst-case Gerzon energy-vector error across keyframes, radians. */
  maxEnergyVectorErrorRad: number;
  updateRateHz: number;
}

export interface SchedulerOptions {
  updateRateHz?: number;
  referenceDistanceM?: number;
  minSourceDistanceM?: number;
}

/** Linear interpolation of the trajectory at time t (edge-held). */
function sampleTrajectory(points: TrajectoryPoint[], timeS: number): TrajectoryPoint {
  if (timeS <= points[0].timeS) {
    return points[0];
  }
  const last = points[points.length - 1];
  if (timeS >= last.timeS) {
    return last;
  }
  let hi = points.findIndex((p) => p.timeS > timeS);
  const a = points[hi - 1];
  const b = points[hi];
  const w = (timeS - a.timeS) / (b.timeS - a.timeS);
  // Interpolate azimuth along the short way around the circle.
  let dAz = b.azimuthRad - a.azimuthRad;
  dAz = Math.atan2(Math.sin(dAz), Math.cos(dAz));
  return {
    timeS,
    azimuthRad: a.azimuthRad + w * dAz,
    distanceM: a.distanceM + w * (b.distanceM - a.distanceM),
  };
}

export function buildGainSchedules(
  trajectory: TrajectoryPoint[],
  acousticMap: AcousticMap,
  options: SchedulerOptions = {},
): ScheduleResult {
  const updateRateHz = options.updateRateHz ?? 50;
  const referenceDistanceM = options.referenceDistanceM ?? 1.0;
  const minSourceDistanceM = options.minSourceDistanceM ?? 0.1;
  if (trajectory.length < 1) {
    throw new Error('trajectory must have at least one point');
  }
  if (updateRateHz <= 0) {
    throw new Error('update rate must be positive');
  }
  for (let i = 1; i < trajectory.length; i += 1) {
    if (trajectory[i].timeS <= trajectory[i - 1].timeS) {
      throw new Error('trajectory times must be strictly increasing');
    }
  }
  for (const p of trajectory) {
    if (p.distanceM <= 0) {
      throw new Error('trajectory distances must be positive');
    }
  }

  const panner = new RingPanner(acousticMap.azimuthsRad);
  const t0 = trajectory[0].timeS;
  const t1 = trajectory[trajectory.length - 1].timeS;
  const tickUs = 1e6 / updateRateHz;
  const tickCount = Math.max(1, Math.ceil((t1 - t0) * updateRateHz) + 1);

  const speakerCount = acousticMap.azimuthsRad.length;
  const schedules: SpeakerSchedule[] = Array.from({ length: speakerCount }, (_, i) => ({
    speakerIndex: i,
    frames: [],
  }));
  let maxError = 0;
  for (let seq = 0; seq < tickCount; seq += 1) {
    const timeS = Math.min(t0 + seq / updateRateHz, t1);
    const point = sampleTrajectory(trajectory, timeS);
    const gains = panner.gains(point.azimuthRad);
    maxError = Math.max(
      maxError,
      energyVectorErrorRad(gains, acousticMap.azimuthsRad, point.azimuthRad),
    );
    const attenuation = Math.min(
      referenceDistanceM / Math.max(point.distanceM, minSourceDistanceM),
      1.0,
    );
    const timestampUs = BigInt(Math.round(seq * tickUs));
    for (let ch = 0; ch < speakerCount; ch += 1) {
      schedules[ch].frames.push({
        seq: seq & 0xffff,
        timestampUs,
        gainQ15: gainToQ15(gains[ch] * attenuation),
      });
    }
  }
  return { schedules, maxEnergyVectorErrorRad: maxError, updateRateHz };
}
