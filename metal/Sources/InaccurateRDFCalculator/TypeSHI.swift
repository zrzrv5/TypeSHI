import Foundation

/// One ranked element hypothesis for a type id.
public struct Candidate { public let z: Int; public let symbol: String; public let prob: Float }

/// Full per-type prediction: ranked top-K candidates plus the conformal set.
public struct TypePrediction {
    public let candidates: [Candidate]       // ranked, length ~topK
    public let conformalSet: [Candidate]     // 90%-coverage RAPS set (empty if disabled)
}

/// End-to-end on-device predictor: Metal descriptors -> CoreML -> composition
/// decode + conformal, mirroring the Python lite path (scripts/predict_lite.py):
///   compute_features (Metal)
///     -> log_probs (CoreML), pooled over `envDraws` env samples
///     -> CompositionPrior.marginals (PMI + charge-neutrality beam)
///     -> RAPS conformal set (temperature-scaled).
public final class TypeSHI {
    private let desc: DescriptorComputer
    private let runner: CoreMLRunner
    private let prior: CompositionPrior
    private let calib: CalibParams

    public init(modelURL: URL) throws {
        self.desc = try DescriptorComputer()
        self.runner = try CoreMLRunner(modelURL: modelURL)
        self.prior = CompositionPrior()
        self.calib = DecodeAssets.calib
    }

    public func predict(_ s: Structure, topK: Int = 5, decode: Bool = true,
                        conformal: Bool = true, envDraws: Int = 4) throws -> [TypePrediction] {
        // Base descriptors (draw 0). rdf/pair_extra/frac/glob are deterministic;
        // only env changes across draws, so recompute env only.
        var d = desc.compute(s, envSeed: 1000)
        let T = d.nTypes
        let draws = runner.usesEnv ? max(1, envDraws) : 1

        // pool model log-probs over env draws (geometric mean == mean of log-probs)
        var sumLogp = [[Double]](repeating: [Double](repeating: 0, count: Desc.nClasses),
                                 count: T)
        for draw in 0..<draws {
            if draw > 0 {
                let (eD, eT) = desc.sampleEnv(s, nTypes: T, seed: UInt64(1000 + draw))
                d.envD = eD; d.envT = eT
            }
            let lp = try runner.logProbs(d)                 // (T, 94)
            for t in 0..<T { for c in 0..<Desc.nClasses { sumLogp[t][c] += Double(lp[t][c]) } }
        }
        let meanLogp = sumLogp.map { row in row.map { $0 / Double(draws) } }
        let frac = d.frac.map { Double($0) }

        // decode -> per-type probability rows
        let probs: [[Double]] = decode
            ? prior.marginals(meanLogp, frac)
            : meanLogp.map { row -> [Double] in
                let m = row.max() ?? 0
                let e = row.map { exp($0 - m) }
                let s = e.reduce(0, +)
                return e.map { $0 / s }
              }

        return (0..<T).map { t in
            let ranked = probs[t].enumerated().sorted { $0.element > $1.element }
            let cands = ranked.prefix(topK).map {
                Candidate(z: $0.offset + 1, symbol: Elements.symbol($0.offset + 1),
                          prob: Float($0.element))
            }
            var setC = [Candidate]()
            if conformal {
                for cls in Conformal.set(probs[t], calib) {
                    setC.append(Candidate(z: cls + 1, symbol: Elements.symbol(cls + 1),
                                          prob: Float(probs[t][cls])))
                }
            }
            return TypePrediction(candidates: Array(cands), conformalSet: setC)
        }
    }
}

func softmax(_ x: [Float]) -> [Float] {
    let m = x.max() ?? 0
    let e = x.map { expf($0 - m) }
    let s = e.reduce(0, +)
    return e.map { $0 / s }
}

public enum Elements {
    static let symbols = ["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg",
        "Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co",
        "Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo",
        "Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La",
        "Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf",
        "Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr",
        "Ra","Ac","Th","Pa","U"]  // Z = 1..94
    public static func symbol(_ z: Int) -> String {
        (z >= 1 && z <= symbols.count) ? symbols[z - 1] : "?"
    }
}
