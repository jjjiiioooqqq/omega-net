/**
 * Pairwise amplitude panner (2D VBAP) over azimuth-sorted speakers.
 * Mirrors engint.spatial_audio.panning (Python) — keep in sync.
 */

const TWO_PI = 2 * Math.PI;

const mod2pi = (a: number): number => ((a % TWO_PI) + TWO_PI) % TWO_PI;

export class RingPanner {
  private readonly order: number[];
  private readonly sorted: number[];
  private readonly count: number;

  constructor(azimuthsRad: number[], minGapRad = 1e-6) {
    if (azimuthsRad.length < 2) {
      throw new Error('panner requires at least 2 azimuths');
    }
    const wrapped = azimuthsRad.map(mod2pi);
    this.order = wrapped
      .map((a, i) => [a, i] as const)
      .sort((p, q) => p[0] - q[0])
      .map(([, i]) => i);
    this.sorted = this.order.map((i) => wrapped[i]);
    this.count = wrapped.length;
    for (let i = 0; i < this.count; i += 1) {
      const next = i + 1 < this.count ? this.sorted[i + 1] : this.sorted[0] + TWO_PI;
      if (next - this.sorted[i] < minGapRad) {
        throw new Error('two speakers share the same azimuth; panning is singular');
      }
    }
  }

  /** Power-normalized gains (sum of squares == 1) for a source azimuth. */
  gains(azimuthRad: number): number[] {
    const target = mod2pi(azimuthRad);
    let lo = this.sorted.findIndex((a) => a > target) - 1;
    if (lo < -1) {
      lo = this.count - 1; // every azimuth <= target
    }
    lo = (lo + this.count) % this.count;
    const hi = (lo + 1) % this.count;

    const aLo = this.sorted[lo];
    const aHi = this.sorted[hi];
    // Solve [cos aLo, cos aHi; sin aLo, sin aHi] g = [cos t; sin t].
    const det = Math.cos(aLo) * Math.sin(aHi) - Math.cos(aHi) * Math.sin(aLo);
    let gLo = (Math.sin(aHi) * Math.cos(target) - Math.cos(aHi) * Math.sin(target)) / det;
    let gHi = (-Math.sin(aLo) * Math.cos(target) + Math.cos(aLo) * Math.sin(target)) / det;
    gLo = Math.max(gLo, 0);
    gHi = Math.max(gHi, 0);
    let norm = Math.hypot(gLo, gHi);
    if (norm === 0) {
      // Target diametrically opposes a wide pair; snap to the nearer edge.
      if (mod2pi(target - aLo) <= mod2pi(aHi - target)) {
        gLo = 1;
      } else {
        gHi = 1;
      }
      norm = 1;
    }
    const out = new Array<number>(this.count).fill(0);
    out[this.order[lo]] = gLo / norm;
    out[this.order[hi]] = gHi / norm;
    return out;
  }
}

/**
 * Angle between the Gerzon energy vector of `gains` and the target
 * direction: the truthful localization metric (0 = energy centroid
 * points exactly at the panned direction).
 */
export function energyVectorErrorRad(
  gains: number[],
  azimuthsRad: number[],
  targetAzimuthRad: number,
): number {
  let ex = 0;
  let ey = 0;
  let total = 0;
  for (let i = 0; i < gains.length; i += 1) {
    const e = gains[i] * gains[i];
    ex += e * Math.cos(azimuthsRad[i]);
    ey += e * Math.sin(azimuthsRad[i]);
    total += e;
  }
  if (total <= 0) {
    throw new Error('gains carry no energy');
  }
  const magnitude = Math.hypot(ex, ey) / total;
  if (magnitude === 0) {
    return Math.PI;
  }
  const cosine =
    (ex * Math.cos(targetAzimuthRad) + ey * Math.sin(targetAzimuthRad)) / (total * magnitude);
  return Math.acos(Math.min(1, Math.max(-1, cosine)));
}
