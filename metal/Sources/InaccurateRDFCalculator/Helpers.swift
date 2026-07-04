import Foundation
import simd

// CPU post-processing that mirrors src/typeid2elem/descriptors.py exactly.
// These are T x T x 64 sized (tiny) — no need to push them to the GPU.

func prefixSum(_ x: [UInt32]) -> [Double] {
    var out = [Double](repeating: 0, count: x.count); var acc = 0.0
    for i in 0..<x.count { acc += Double(x[i]); out[i] = acc }
    return out
}

func argmax(_ x: [Float]) -> (Int, Float) {
    var bi = 0; var bv = -Float.greatestFiniteMagnitude
    for i in 0..<x.count where x[i] > bv { bv = x[i]; bi = i }
    return (bi, bv)
}

/// Ideal-gas-normalized, Gaussian-smeared g_ab(r), clipped to 1e4.
/// Order matches Python: smear the RAW histogram, then divide by
/// counts_a * rho_b * shell_volume.
func normalizeRDF(_ raw: [UInt32], centers: Double, rhoB: Double) -> [Float] {
    let nb = Desc.nBins
    let smeared = gaussianSmooth(raw.map { Double($0) }, sigma: Double(Desc.smearBins))
    var g = [Float](repeating: 0, count: nb)
    let denomCommon = max(centers, 1) * max(rhoB, 1e-12)
    for k in 0..<nb {
        let rLo = Double(k) * Double(Desc.dr)
        let rHi = Double(k + 1) * Double(Desc.dr)
        let shell = 4.0 / 3.0 * .pi * (rHi * rHi * rHi - rLo * rLo * rLo)
        let denom = denomCommon * shell
        g[k] = Float(min(smeared[k] / max(denom, 1e-12), 1e4))
    }
    return g
}

/// 1-D Gaussian filter, sigma in bins, 'constant' (zero) boundary — matches
/// scipy.ndimage.gaussian_filter1d(..., mode="constant").
func gaussianSmooth(_ x: [Double], sigma: Double) -> [Double] {
    if sigma <= 0 { return x }
    let radius = Int((4.0 * sigma).rounded())
    var kernel = [Double](repeating: 0, count: 2 * radius + 1)
    var norm = 0.0
    for i in -radius...radius {
        let w = exp(-0.5 * Double(i * i) / (sigma * sigma))
        kernel[i + radius] = w; norm += w
    }
    for i in 0..<kernel.count { kernel[i] /= norm }
    var out = [Double](repeating: 0, count: x.count)
    for k in 0..<x.count {
        var acc = 0.0
        for j in -radius...radius {
            let idx = k + j
            if idx >= 0 && idx < x.count { acc += x[idx] * kernel[j + radius] }
        }
        out[k] = acc
    }
    return out
}

/// median and 10th percentile of per-center nearest distance (atoms of type a
/// to type b). numpy percentile uses linear interpolation.
func nnStats(_ nnDist: [Float], nTypes T: Int, typeA a: Int, typeB b: Int,
             types: [UInt32]) -> (Float, Float) {
    var vals = [Float]()
    for i in 0..<types.count where Int(types[i]) == a {
        vals.append(nnDist[i * T + b])
    }
    if vals.isEmpty { return (Desc.rMax, Desc.rMax) }
    vals.sort()
    return (percentile(vals, 50), percentile(vals, 10))
}

func percentile(_ sorted: [Float], _ q: Float) -> Float {
    if sorted.count == 1 { return sorted[0] }
    let pos = q / 100 * Float(sorted.count - 1)
    let lo = Int(pos.rounded(.down)); let hi = Int(pos.rounded(.up))
    let frac = pos - Float(lo)
    return sorted[lo] * (1 - frac) + sorted[hi] * frac
}

/// Cell-less vacuum box volume, matching the descriptors.py convention:
/// span = (max - min) + 10 A; volume = prod(span).
func openBoxVolume(_ pos: [SIMD3<Float>]) -> Float {
    var lo = pos[0], hi = pos[0]
    for p in pos { lo = simd_min(lo, p); hi = simd_max(hi, p) }
    let span = (hi - lo) + SIMD3<Float>(repeating: 10)
    return span.x * span.y * span.z
}
