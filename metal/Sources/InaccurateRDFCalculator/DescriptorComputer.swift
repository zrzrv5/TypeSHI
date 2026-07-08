import Foundation
import Metal
import simd

/// The descriptor set that feeds the CoreML model. Layout mirrors
/// `typeid2elem.descriptors.compute_features` exactly (padding to `Desc.maxTypes`
/// happens later, in `CoreMLRunner`). All arrays are row-major.
public struct Descriptors {
    public let nTypes: Int
    public var rdf: [Float]         // nTypes * nTypes * nBins       (a<-b)
    public var pairExtra: [Float]   // nTypes * nTypes * nPairExtra
    public var frac: [Float]        // nTypes
    public var glob: [Float]        // nGlob
    public var envD: [Float]        // nTypes * mEnv * kEnv
    public var envT: [Float]        // nTypes * mEnv * kEnv  (-1 pad; float for CoreML)
}

public struct Structure {
    public var positions: [SIMD3<Float>]
    public var typeIds: [UInt32]        // 0-based contiguous
    public var cell: (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)?   // nil = open box
    public init(positions: [SIMD3<Float>], typeIds: [UInt32],
                cell: (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)?) {
        self.positions = positions; self.typeIds = typeIds; self.cell = cell
    }
}

/// Deterministic splitmix64 — used only to sample the M_ENV env centers per type.
/// Env is a high-variance sampled feature and the golden env is a structural check
/// only, so the exact RNG need not match numpy; it only has to be reproducible and
/// to vary across the pooled draws.
struct SplitMix64 {
    var state: UInt64
    init(_ seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state = state &+ 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    /// pick `k` distinct indices from `pool` (partial Fisher–Yates); returns all if k >= count.
    mutating func sample(_ pool: [Int], _ k: Int) -> [Int] {
        if pool.count <= k { return pool }
        var a = pool
        for i in 0..<k {
            let j = i + Int(next() % UInt64(a.count - i))
            a.swapAt(i, j)
        }
        return Array(a[0..<k])
    }
}

/// GPU descriptor front-end. Emits the heavy histogram + nearest aggregates and
/// the env sets on the Metal device, then finishes the T×T-sized post-processing
/// on the CPU.
public final class DescriptorComputer {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let rdfPipeline: MTLComputePipelineState
    private let envPipeline: MTLComputePipelineState

    public init() throws {
        guard let dev = MTLCreateSystemDefaultDevice(),
              let q = dev.makeCommandQueue() else { throw Err.noDevice }
        self.device = dev
        self.queue = q
        // SwiftPM copies Shaders.metal into the bundle as a resource rather than
        // compiling a default.metallib, so compile the source at runtime. This is
        // portable across SwiftPM quirks and works on-device (macOS + iOS).
        guard let url = Bundle.module.url(forResource: "Shaders", withExtension: "metal") else {
            throw Err.noShaderSource
        }
        let src = try String(contentsOf: url, encoding: .utf8)
        let lib = try dev.makeLibrary(source: src, options: nil)
        self.rdfPipeline = try dev.makeComputePipelineState(
            function: lib.makeFunction(name: "rdfHistogram")!)
        self.envPipeline = try dev.makeComputePipelineState(
            function: lib.makeFunction(name: "envSample")!)
    }

    enum Err: Error { case noDevice, noShaderSource }

    // Params mirrored from Shaders.metal — scalar-only, identical field order.
    private struct Params {
        var rMax: Float; var dr: Float; var rEnv: Float
        var nBins: UInt32; var nTypes: UInt32; var nAtoms: UInt32
        var nx: Int32; var ny: Int32; var nz: Int32
        var pbc: UInt32
    }

    private func makeParams(_ s: Structure, nTypes T: Int) -> (Params, [SIMD3<Float>]) {
        let n = s.positions.count
        var nx: Int32 = 0, ny: Int32 = 0, nz: Int32 = 0
        var rows = [SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 0),
                    SIMD3<Float>(repeating: 0)]
        if let c = s.cell {
            rows = [c.0, c.1, c.2]
            let vol = abs(simd_determinant(simd_float3x3(c.0, c.1, c.2)))
            // perpendicular height along axis i = vol / |a_{i+1} x a_{i+2}|;
            // n_i = ceil(rMax / height_i) covers every image within the cutoff.
            func reps(_ p: SIMD3<Float>, _ q: SIMD3<Float>) -> Int32 {
                let area = simd_length(simd_cross(p, q))
                let h = area > 0 ? vol / area : Float.greatestFiniteMagnitude
                return Int32(max(1, Int((Desc.rMax / h).rounded(.up))))
            }
            nx = reps(c.1, c.2); ny = reps(c.2, c.0); nz = reps(c.0, c.1)
        }
        let p = Params(rMax: Desc.rMax, dr: Desc.dr, rEnv: Desc.rEnv,
                       nBins: UInt32(Desc.nBins), nTypes: UInt32(T), nAtoms: UInt32(n),
                       nx: nx, ny: ny, nz: nz, pbc: s.cell == nil ? 0 : 1)
        return (p, rows)
    }

    /// Full descriptor set for one env draw. `envSeed` picks the env centers.
    public func compute(_ s: Structure, envSeed: UInt64 = 0) -> Descriptors {
        let n = s.positions.count
        let T = Int(s.typeIds.max().map { $0 + 1 } ?? 0)
        let nb = Desc.nBins

        let (hist, nnDist) = runHistogram(s, nTypes: T)

        // counts per type (host)
        var counts = [Double](repeating: 0, count: T)
        for t in s.typeIds { counts[Int(t)] += 1 }

        // volume / density. NOTE: has_cell is ALWAYS 1.0 in compute_features
        // (the cell-less path uses a padded vacuum box treated as periodic), so
        // the model never saw 0 — emit 1.0 regardless.
        let hasCell: Float = 1.0
        let volume = s.cell.map { abs(simd_determinant(simd_float3x3($0.0, $0.1, $0.2))) }
            ?? openBoxVolume(s.positions)
        let rhoB = counts.map { $0 / Double(volume) }   // per-type number density

        var rdf = [Float](repeating: 0, count: T * T * nb)
        var pairExtra = [Float](repeating: 0, count: T * T * Desc.nPairExtra)

        for a in 0..<T {
            for b in 0..<T {
                let base = (a * T + b) * nb
                let raw = Array(hist[base..<base + nb])            // counts
                // running coordination numbers BEFORE smearing (physical counts)
                let cum = prefixSum(raw).map { $0 / max(counts[a], 1) }
                let cn = Desc.cnRadii.map { r -> Float in
                    Float(cum[min(Int(r / Desc.dr), nb - 1)])
                }
                // ideal-gas-normalized, smeared g_ab(r); clip <= 1e4
                let g = normalizeRDF(raw, centers: counts[a], rhoB: rhoB[b])
                for k in 0..<nb { rdf[base + k] = g[k] }

                let (peakBin, peakVal) = argmax(g)
                let peakPos = (Float(peakBin) + 0.5) * Desc.dr / Desc.rMax
                let peakH = log1pf(peakVal)

                let (nnMed, nnP10) = nnStats(nnDist, nTypes: T, typeA: a, typeB: b,
                                             types: s.typeIds)

                let pe = (a * T + b) * Desc.nPairExtra
                for (k, v) in cn.enumerated() { pairExtra[pe + k] = log1pf(v) }
                pairExtra[pe + 4] = nnMed / Desc.rMax
                pairExtra[pe + 5] = nnP10 / Desc.rMax
                pairExtra[pe + 6] = peakPos
                pairExtra[pe + 7] = peakH
            }
        }

        let frac = counts.map { Float($0 / Double(n)) }
        let glob: [Float] = [logf(Float(Double(n) / Double(volume))),
                             hasCell, 1.0 / Float(T)]

        let (envD, envT) = sampleEnv(s, nTypes: T, seed: envSeed)
        return Descriptors(nTypes: T, rdf: rdf, pairExtra: pairExtra,
                           frac: frac, glob: glob, envD: envD, envT: envT)
    }

    // MARK: - GPU dispatch: histogram + per-center nearest distance

    private func runHistogram(_ s: Structure, nTypes T: Int)
        -> (hist: [UInt32], nnDist: [Float]) {
        let n = s.positions.count
        let nb = Desc.nBins
        let histCount = T * T * nb

        let posBuf = device.makeBuffer(bytes: s.positions,
            length: MemoryLayout<SIMD3<Float>>.stride * n)!
        let typeBuf = device.makeBuffer(bytes: s.typeIds,
            length: MemoryLayout<UInt32>.stride * n)!
        let histBuf = device.makeBuffer(length: MemoryLayout<UInt32>.stride * histCount,
            options: .storageModeShared)!
        memset(histBuf.contents(), 0, histBuf.length)
        // nnDist holds float bits; the kernel does an unsigned atomic-min on them.
        var nn = [Float](repeating: Desc.rMax, count: n * T)
        let nnBuf = device.makeBuffer(bytes: &nn,
            length: MemoryLayout<Float>.stride * n * T, options: .storageModeShared)!

        var (p, rows) = makeParams(s, nTypes: T)
        let paramBuf = device.makeBuffer(bytes: &p, length: MemoryLayout<Params>.stride)!
        let cellBuf = device.makeBuffer(bytes: rows,
            length: MemoryLayout<SIMD3<Float>>.stride * 3)!

        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(rdfPipeline)
        enc.setBuffer(posBuf, offset: 0, index: 0)
        enc.setBuffer(typeBuf, offset: 0, index: 1)
        enc.setBuffer(histBuf, offset: 0, index: 2)
        enc.setBuffer(nnBuf, offset: 0, index: 3)
        enc.setBuffer(paramBuf, offset: 0, index: 4)
        enc.setBuffer(cellBuf, offset: 0, index: 5)
        let w = rdfPipeline.maxTotalThreadsPerThreadgroup
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(w, n),
                                                           height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        let hist = Array(UnsafeBufferPointer(
            start: histBuf.contents().assumingMemoryBound(to: UInt32.self),
            count: histCount))
        let nnOut = Array(UnsafeBufferPointer(
            start: nnBuf.contents().assumingMemoryBound(to: Float.self),
            count: n * T))
        return (hist, nnOut)
    }

    // MARK: - GPU dispatch: env sets (per type, M sampled atoms x K nearest)

    /// Resample only the env sets for a given seed (rdf/pair_extra/frac/glob are
    /// deterministic, so pooling draws only varies env). Returns (envD, envT).
    public func sampleEnv(_ s: Structure, nTypes T: Int, seed: UInt64)
        -> (envD: [Float], envT: [Float]) {
        let n = s.positions.count
        let mk = Desc.mEnv * Desc.kEnv
        var envD = [Float](repeating: 0, count: T * mk)
        var envT = [Float](repeating: -1, count: T * mk)
        if n == 0 || T == 0 { return (envD, envT) }

        // host bucketing: up to M_ENV atoms per type, seeded sample without replacement
        var byType = [[Int]](repeating: [], count: T)
        for (i, t) in s.typeIds.enumerated() { byType[Int(t)].append(i) }
        var rng = SplitMix64(seed &+ 0x1234_5678)
        var chosenAtom = [UInt32](), chosenType = [UInt32](), chosenSlot = [UInt32]()
        for a in 0..<T {
            let picks = rng.sample(byType[a], Desc.mEnv)
            for (m, atom) in picks.enumerated() {
                chosenAtom.append(UInt32(atom))
                chosenType.append(UInt32(a))
                chosenSlot.append(UInt32(m))
            }
        }
        let nChosen = chosenAtom.count
        if nChosen == 0 { return (envD, envT) }

        let posBuf = device.makeBuffer(bytes: s.positions,
            length: MemoryLayout<SIMD3<Float>>.stride * n)!
        let typeBuf = device.makeBuffer(bytes: s.typeIds,
            length: MemoryLayout<UInt32>.stride * n)!
        let caBuf = device.makeBuffer(bytes: chosenAtom,
            length: MemoryLayout<UInt32>.stride * nChosen)!
        let ctBuf = device.makeBuffer(bytes: chosenType,
            length: MemoryLayout<UInt32>.stride * nChosen)!
        let csBuf = device.makeBuffer(bytes: chosenSlot,
            length: MemoryLayout<UInt32>.stride * nChosen)!
        var (p, rows) = makeParams(s, nTypes: T)
        let paramBuf = device.makeBuffer(bytes: &p, length: MemoryLayout<Params>.stride)!
        let cellBuf = device.makeBuffer(bytes: rows,
            length: MemoryLayout<SIMD3<Float>>.stride * 3)!
        var nChosenU = UInt32(nChosen), mEnvU = UInt32(Desc.mEnv), kEnvU = UInt32(Desc.kEnv)
        let nchBuf = device.makeBuffer(bytes: &nChosenU, length: 4)!
        let mBuf = device.makeBuffer(bytes: &mEnvU, length: 4)!
        let kBuf = device.makeBuffer(bytes: &kEnvU, length: 4)!
        let envDBuf = device.makeBuffer(bytes: &envD,
            length: MemoryLayout<Float>.stride * envD.count, options: .storageModeShared)!
        let envTBuf = device.makeBuffer(bytes: &envT,
            length: MemoryLayout<Float>.stride * envT.count, options: .storageModeShared)!

        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(envPipeline)
        enc.setBuffer(posBuf, offset: 0, index: 0)
        enc.setBuffer(typeBuf, offset: 0, index: 1)
        enc.setBuffer(caBuf, offset: 0, index: 2)
        enc.setBuffer(ctBuf, offset: 0, index: 3)
        enc.setBuffer(csBuf, offset: 0, index: 4)
        enc.setBuffer(paramBuf, offset: 0, index: 5)
        enc.setBuffer(cellBuf, offset: 0, index: 6)
        enc.setBuffer(nchBuf, offset: 0, index: 7)
        enc.setBuffer(mBuf, offset: 0, index: 8)
        enc.setBuffer(kBuf, offset: 0, index: 9)
        enc.setBuffer(envDBuf, offset: 0, index: 10)
        enc.setBuffer(envTBuf, offset: 0, index: 11)
        let w = envPipeline.maxTotalThreadsPerThreadgroup
        enc.dispatchThreads(MTLSize(width: nChosen, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(w, nChosen),
                                                           height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        envD = Array(UnsafeBufferPointer(
            start: envDBuf.contents().assumingMemoryBound(to: Float.self), count: envD.count))
        envT = Array(UnsafeBufferPointer(
            start: envTBuf.contents().assumingMemoryBound(to: Float.self), count: envT.count))
        return (envD, envT)
    }
}
