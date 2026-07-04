import Foundation

/// One ranked element hypothesis for a type id.
public struct Candidate { public let z: Int; public let symbol: String; public let prob: Float }

/// End-to-end on-device predictor: Metal descriptors -> CoreML -> ranked elements.
///
/// On-device this stops at per-type softmax top-K. The composition decode
/// (PMI + charge-neutrality beam) and RAPS conformal sets are cheap CPU steps that
/// currently live in Python (`decode.py`, `calibrate.py`) — port them here when the
/// descriptor path is trusted; the arrays they need (`costats.npz`, `calib.npz`)
/// ship in `weights/`.  TODO: env 4-draw pooling once envSample lands.
public final class TypeSHI {
    private let desc: DescriptorComputer
    private let runner: CoreMLRunner

    public init(modelURL: URL) throws {
        self.desc = try DescriptorComputer()
        self.runner = try CoreMLRunner(modelURL: modelURL)
    }

    public func predict(_ s: Structure, topK: Int = 5) throws -> [[Candidate]] {
        let d = desc.compute(s)
        let logp = try runner.logProbs(d)
        return logp.map { row in
            let probs = softmax(row)
            return probs.enumerated()
                .sorted { $0.element > $1.element }
                .prefix(topK)
                .map { Candidate(z: $0.offset + 1,
                                 symbol: Elements.symbol($0.offset + 1),
                                 prob: $0.element) }
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
