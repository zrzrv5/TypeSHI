import Foundation
import Metal
import simd

/// The descriptor set that feeds the CoreML model. Layout mirrors
/// `typeid2elem.descriptors.compute_features` exactly (padded to `Desc.maxTypes`
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

/// GPU descriptor front-end. Emits the heavy histogram + nearest aggregates on the
/// Metal device, then finishes the T×T-sized post-processing on the CPU.
public final class DescriptorComputer {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let rdfPipeline: MTLComputePipelineState

    public init() throws {
        guard let dev = MTLCreateSystemDefaultDevice(),
              let q = dev.makeCommandQueue() else { throw Err.noDevice }
        self.device = dev
        self.queue = q
        let lib = try dev.makeDefaultLibrary(bundle: .module)
        self.rdfPipeline = try dev.makeComputePipelineState(
            function: lib.makeFunction(name: "rdfHistogram")!)
    }

    enum Err: Error { case noDevice }

    public func compute(_ s: Structure) -> Descriptors {
        let n = s.positions.count
        let T = Int(s.typeIds.max().map { $0 + 1 } ?? 0)
        let nb = Desc.nBins

        // --- GPU: histogram + per-center nearest distance to each type ---
        let (hist, nnDist) = runHistogram(s, nTypes: T)

        // counts per type (host)
        var counts = [Double](repeating: 0, count: T)
        for t in s.typeIds { counts[Int(t)] += 1 }

        // volume / density
        let hasCell: Float = s.cell == nil ? 0 : 1
        let volume = s.cell.map { abs(simd_determinant(simd_float3x3($0.0, $0.1, $0.2))) }
            ?? openBoxVolume(s.positions)
        let rhoB = counts.map { $0 / Double(volume) }   // per-type number density

        // --- CPU: normalize -> g_ab(r), CN, peak, robust NN, frac, glob ---
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

                // robust NN stats over centers of type a (from nnDist[i, b])
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

        // --- env sets (TODO: GPU sampler; zeros/-1 pad for now) ---
        let envCount = T * Desc.mEnv * Desc.kEnv
        let env = Descriptors(
            nTypes: T, rdf: rdf, pairExtra: pairExtra, frac: frac, glob: glob,
            envD: [Float](repeating: 0, count: envCount),
            envT: [Float](repeating: -1, count: envCount))
        // TODO: fill env via the envSample kernel + 4-draw pooling downstream.
        return env
    }

    // MARK: - GPU dispatch

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
        var nn = [Float](repeating: Desc.rMax, count: n * T)
        let nnBuf = device.makeBuffer(bytes: &nn,
            length: MemoryLayout<Float>.stride * n * T, options: .storageModeShared)!

        var p = makeParams(s, nTypes: T)
        let paramBuf = device.makeBuffer(bytes: &p,
            length: MemoryLayout<Params>.stride)!

        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(rdfPipeline)
        enc.setBuffer(posBuf, offset: 0, index: 0)
        enc.setBuffer(typeBuf, offset: 0, index: 1)
        enc.setBuffer(histBuf, offset: 0, index: 2)
        enc.setBuffer(nnBuf, offset: 0, index: 3)
        enc.setBuffer(paramBuf, offset: 0, index: 4)
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

    // Params struct mirrored from Shaders.metal (keep field order identical).
    private struct Params {
        var rMax: Float; var dr: Float
        var nBins: UInt32; var nTypes: UInt32; var nAtoms: UInt32
        var cell0: SIMD3<Float>; var cell1: SIMD3<Float>; var cell2: SIMD3<Float>
        var pbc: UInt32
    }

    private func makeParams(_ s: Structure, nTypes T: Int) -> Params {
        let c = s.cell ?? (SIMD3(0, 0, 0), SIMD3(0, 0, 0), SIMD3(0, 0, 0))
        return Params(rMax: Desc.rMax, dr: Desc.dr,
                      nBins: UInt32(Desc.nBins), nTypes: UInt32(T),
                      nAtoms: UInt32(s.positions.count),
                      cell0: c.0, cell1: c.1, cell2: c.2,
                      pbc: s.cell == nil ? 0 : 1)
    }
}
