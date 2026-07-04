import CoreML
import Foundation

/// Runs the exported TypeSHI model (env_codfull_sharp30.mlpackage) on descriptors.
/// Inputs/outputs mirror scripts/export_model.py:
///   rdf (1,T,T,64) · pair_extra (1,T,T,8) · frac (1,T) · glob (1,3) ·
///   mask (1,T float) · env_d (1,T,16,16) · env_t (1,T,16,16 float, -1 pad)
///   -> log_probs (1,T,94).  T is the fixed exported dim (Desc.maxTypes); real
///   types occupy the first nTypes rows, the rest are masked padding.
public final class CoreMLRunner {
    private let model: MLModel
    private let hasEnv: Bool

    public init(modelURL: URL) throws {
        // Accepts a compiled .mlmodelc or an .mlpackage (compiled on first load).
        let compiled = modelURL.pathExtension == "mlpackage"
            ? try MLModel.compileModel(at: modelURL) : modelURL
        self.model = try MLModel(contentsOf: compiled)
        self.hasEnv = model.modelDescription.inputDescriptionsByName["env_d"] != nil
    }

    /// Returns log-probs (nTypes x nClasses). Pool 4 env draws upstream and
    /// average these when env is active (matches the Python inference default).
    public func logProbs(_ d: Descriptors) throws -> [[Float]] {
        let T = Desc.maxTypes, nb = Desc.nBins, npx = Desc.nPairExtra
        let n = d.nTypes

        let rdf = try pad4(d.rdf, n: n, T: T, d1: T, d2: nb, srcT2: n)
        let pe = try pad4(d.pairExtra, n: n, T: T, d1: T, d2: npx, srcT2: n)
        let frac = try padVec(d.frac, n: n, T: T)
        let glob = try array([1, Desc.nGlob], d.glob)
        let mask = try maskVec(n: n, T: T)

        var feats: [String: MLFeatureValue] = [
            "rdf": .init(multiArray: rdf), "pair_extra": .init(multiArray: pe),
            "frac": .init(multiArray: frac), "glob": .init(multiArray: glob),
            "mask": .init(multiArray: mask),
        ]
        if hasEnv {
            feats["env_d"] = .init(multiArray:
                try padEnv(d.envD, n: n, T: T, fill: 0))
            feats["env_t"] = .init(multiArray:
                try padEnv(d.envT, n: n, T: T, fill: -1))
        }
        let out = try model.prediction(
            from: MLDictionaryFeatureProvider(dictionary: feats))
        guard let lp = out.featureValue(for: "log_probs")?.multiArrayValue else {
            throw NSError(domain: "TypeSHI", code: 1)
        }
        // slice (1,T,94) -> first n rows
        var result = [[Float]]()
        for t in 0..<n {
            var row = [Float](repeating: 0, count: Desc.nClasses)
            for c in 0..<Desc.nClasses {
                row[c] = lp[[0, t, c] as [NSNumber]].floatValue
            }
            result.append(row)
        }
        return result
    }

    // MARK: - MLMultiArray builders (zero-padded to the fixed T)

    private func array(_ shape: [Int], _ data: [Float]) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: shape as [NSNumber], dataType: .float32)
        let p = a.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<min(data.count, a.count) { p[i] = data[i] }
        return a
    }

    private func pad4(_ src: [Float], n: Int, T: Int, d1: Int, d2: Int,
                      srcT2: Int) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: [1, T, d1, d2] as [NSNumber],
                                 dataType: .float32)
        let p = a.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<a.count { p[i] = 0 }
        for i in 0..<n { for j in 0..<n { for k in 0..<d2 {
            let dst = ((i * T) + j) * d2 + k
            let s = ((i * srcT2) + j) * d2 + k
            p[dst] = src[s]
        }}}
        return a
    }

    private func padVec(_ src: [Float], n: Int, T: Int) throws -> MLMultiArray {
        try array([1, T], (0..<T).map { $0 < n ? src[$0] : 0 })
    }
    private func maskVec(n: Int, T: Int) throws -> MLMultiArray {
        try array([1, T], (0..<T).map { $0 < n ? 1 : 0 })
    }
    private func padEnv(_ src: [Float], n: Int, T: Int, fill: Float) throws
        -> MLMultiArray {
        let mk = Desc.mEnv * Desc.kEnv
        var buf = [Float](repeating: fill, count: T * mk)
        for i in 0..<(n * mk) { buf[i] = src[i] }
        return try array([1, T, Desc.mEnv, Desc.kEnv], buf)
    }
}
