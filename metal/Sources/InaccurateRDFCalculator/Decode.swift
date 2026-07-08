import Foundation

// Composition-aware joint decoding + RAPS conformal sets — a faithful port of
// src/typeid2elem/decode.py (CompositionPrior) and the conformal block of
// scripts/predict_lite.py. All math is done in Double to match numpy.
//
// The PMI matrix and calibration scalars are baked into decode_assets.json by
// metal/tools/gen_decode_assets.py (from weights/costats.npz + weights/calib.npz).

public struct CalibParams {
    public let temperature: Double
    public let qhat: Double
    public let lam: Double
    public let kReg: Int
    public let coverage: Double
}

/// Loads the bundled decode/conformal assets (PMI matrix + calibration scalars).
/// `_loaded` is a lazily-initialized static — parsed once on first use.
public enum DecodeAssets {
    public static let nEl = 94

    static func load() -> (pmi: [Double], calib: CalibParams) {
        guard let url = Bundle.module.url(forResource: "decode_assets",
                                          withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let raw = try? JSONSerialization.jsonObject(with: data),
              let obj = raw as? [String: Any],
              let pmiAny = obj["pmi"] as? [Any],
              let cal = obj["calib"] as? [String: Any] else {
            return ([Double](repeating: 0, count: 94 * 94),
                    CalibParams(temperature: 1, qhat: 1, lam: 0, kReg: 0, coverage: 0.9))
        }
        let pmi = pmiAny.map { ($0 as? NSNumber)?.doubleValue ?? 0 }
        func d(_ k: String) -> Double { ((cal[k] as? NSNumber)?.doubleValue ?? 0) }
        let calib = CalibParams(temperature: d("temperature"), qhat: d("qhat"),
                                lam: d("lam"), kReg: Int(d("k_reg")), coverage: d("coverage"))
        return (pmi, calib)
    }

    public static let _loaded = load()
    public static var calib: CalibParams { _loaded.calib }
}

/// Composition prior: re-ranks joint element assignments (one element per type
/// id) by model log-prob + PMI(co-occurrence) + charge-neutrality feasibility.
public final class CompositionPrior {
    public let pmi: [Double]                // (94*94)
    let wPmi: Double
    let wNeut: Double
    let nEl = 94

    public init(wPmi: Double = 0.5, wNeut: Double = 1.5) {
        self.pmi = DecodeAssets._loaded.pmi
        self.wPmi = wPmi
        self.wNeut = wNeut
    }

    /// Min |Σ_t x_t q_t| over allowed oxidation states; 0 => neutral achievable.
    private func neutrality(_ zs: [Int], _ frac: [Double]) -> Double {
        let options = zs.map { OXIDATION[$0] ?? [0] }
        var idx = [Int](repeating: 0, count: zs.count)
        var best = Double.greatestFiniteMagnitude
        while true {
            var dot = 0.0
            for t in 0..<zs.count { dot += frac[t] * Double(options[t][idx[t]]) }
            best = min(best, abs(dot))
            if best < 1e-9 { return 0.0 }
            // odometer increment
            var pos = zs.count - 1
            while pos >= 0 {
                idx[pos] += 1
                if idx[pos] < options[pos].count { break }
                idx[pos] = 0; pos -= 1
            }
            if pos < 0 { break }
        }
        return best
    }

    private func scoreExtra(_ zs: [Int], _ frac: [Double]) -> Double {
        let els = Array(Set(zs)).sorted()
        var pmiVal = 0.0
        if els.count > 1 {
            var acc = 0.0; var cnt = 0
            for i in 0..<els.count {
                for j in (i + 1)..<els.count {
                    acc += pmi[(els[i] - 1) * nEl + (els[j] - 1)]; cnt += 1
                }
            }
            pmiVal = acc / Double(cnt)
        }
        return wPmi * pmiVal - wNeut * neutrality(zs, frac)
    }

    /// logp: (T, 94) per-type log-probs. Returns scored assignments (desc).
    func rerank(_ logp: [[Double]], _ frac: [Double],
                topK: Int = 10, beam: Int = 300) -> [(Double, [Int])] {
        let T = logp.count
        // per-type Z candidates: top-K by log-prob
        var cand = [[Int]]()
        for t in 0..<T {
            let order = (0..<nEl).sorted { logp[t][$0] > logp[t][$1] }
            cand.append(order.prefix(topK).map { $0 + 1 })
        }
        var partials: [(Double, [Int])] = [(0.0, [])]
        for t in 0..<T {
            var nxt = [(Double, [Int])]()
            nxt.reserveCapacity(partials.count * cand[t].count)
            for (s, zs) in partials {
                for z in cand[t] { nxt.append((s + logp[t][z - 1], zs + [z])) }
            }
            nxt.sort { $0.0 > $1.0 }
            if nxt.count > beam { nxt.removeLast(nxt.count - beam) }
            partials = nxt
        }
        var scored = partials.map { (s, zs) in (s + scoreExtra(zs, frac), zs) }
        scored.sort { $0.0 > $1.0 }
        return scored
    }

    /// Per-type marginal probs (T, 94) from softmax over re-ranked assignments.
    public func marginals(_ logp: [[Double]], _ frac: [Double],
                          topK: Int = 10, beam: Int = 300) -> [[Double]] {
        let T = logp.count
        let scored = rerank(logp, frac, topK: topK, beam: beam)
        let maxS = scored.map { $0.0 }.max() ?? 0
        var w = scored.map { exp($0.0 - maxS) }
        let wsum = w.reduce(0, +)
        for i in 0..<w.count { w[i] /= wsum }
        var out = [[Double]](repeating: [Double](repeating: 1e-12, count: nEl), count: T)
        for (wi, (_, zs)) in zip(w, scored) {
            for (t, z) in zs.enumerated() { out[t][z - 1] += wi }
        }
        for t in 0..<T {
            let rowSum = out[t].reduce(0, +)
            for c in 0..<nEl { out[t][c] /= rowSum }
        }
        return out
    }
}

/// RAPS conformal prediction set + temperature scaling (predict_lite.raps_set).
public enum Conformal {
    /// Returns the conformal set (0-based class ids = Z-1) for one type's decoded
    /// probability vector, applying temperature scaling first.
    public static func set(_ probs: [Double], _ calib: CalibParams) -> [Int] {
        // temperature scaling on log-probs, then renormalize
        var z = probs.map { log($0 + 1e-12) / calib.temperature }
        let m = z.max() ?? 0
        for i in 0..<z.count { z[i] -= m }
        var e = z.map { exp($0) }
        let es = e.reduce(0, +)
        for i in 0..<e.count { e[i] /= es }
        // RAPS: greedily accumulate mass in descending prob order
        let order = (0..<e.count).sorted { e[$0] > e[$1] }
        var cum = 0.0
        var out = [Int]()
        for (i, cls) in order.enumerated() {
            cum += e[cls]
            out.append(cls)
            if cum + calib.lam * Double(max(0, i + 1 - calib.kReg)) >= calib.qhat { break }
        }
        return out
    }
}
