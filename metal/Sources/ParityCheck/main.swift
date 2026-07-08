import Foundation
import InaccurateRDFCalculator
import simd

// Parity harness for the Metal port. Reproduces metal/parity/golden.json with the
// Metal + Swift path and diffs against the Python reference.
//
//   swift run parity-check [golden.json] [--dump out.json] [--model model.mlpackage]
//
// Checks, in order:
//   1. descriptors   rdf / pair_extra / frac / glob   (deterministic; hard pass)
//   2. env sets      structural vs golden.env_seed0   (sampled; informational)
//   3. decode+conf   Swift CompositionPrior + RAPS on golden.log_probs vs
//                    metal/parity/decode_golden.json  (deterministic; hard pass)
//   4. model top-1   only with --model (needs the CoreML .mlpackage)
// --dump writes the Metal descriptors as JSON for the ONNX bridge check.

let args = CommandLine.arguments
var positional = [String]()
var dumpPath: String? = nil
var modelPath: String? = nil
var i = 1
while i < args.count {
    switch args[i] {
    case "--dump": i += 1; dumpPath = i < args.count ? args[i] : nil
    case "--model": i += 1; modelPath = i < args.count ? args[i] : nil
    default: positional.append(args[i])
    }
    i += 1
}
let goldenPath = positional.first ?? "metal/parity/golden.json"

let raw = try Data(contentsOf: URL(fileURLWithPath: goldenPath))
let json = try JSONSerialization.jsonObject(with: raw) as! [String: Any]

let input = json["input"] as! [String: Any]
let posArr = input["positions"] as! [[Double]]
let typeArr = input["type_ids"] as! [Int]
let cellArr = input["cell"] as! [[Double]]

let positions = posArr.map { SIMD3<Float>(Float($0[0]), Float($0[1]), Float($0[2])) }
let typeIds = typeArr.map { UInt32($0) }
let cell = (SIMD3<Float>(cellArr[0].map(Float.init)),
            SIMD3<Float>(cellArr[1].map(Float.init)),
            SIMD3<Float>(cellArr[2].map(Float.init)))
let structure = Structure(positions: positions, typeIds: typeIds, cell: cell)

let expected = json["expected"] as! [String: Any]
func flat(_ dict: [String: Any], _ key: String) -> [Float] {
    func rec(_ v: Any) -> [Float] {
        if let d = v as? Double { return [Float(d)] }
        if let i = v as? Int { return [Float(i)] }
        if let a = v as? [Any] { return a.flatMap(rec) }
        return []
    }
    return rec(dict[key]!)
}

let comp = try DescriptorComputer()
let d = comp.compute(structure)   // envSeed 0

func report(_ name: String, _ got: [Float], _ ref: [Float], tol: Float) -> Bool {
    guard got.count == ref.count else {
        print("✗ \(name): shape \(got.count) vs \(ref.count)"); return false
    }
    var maxAbs: Float = 0
    for i in 0..<got.count { maxAbs = max(maxAbs, abs(got[i] - ref[i])) }
    let ok = maxAbs <= tol
    print("\(ok ? "✓" : "✗") \(name): max|Δ| = \(maxAbs)  (tol \(tol))")
    return ok
}

var allPass = true
print("structure: \(positions.count) atoms, \(d.nTypes) types\n")
print("[1] descriptors")
allPass = report("rdf",        d.rdf,       flat(expected, "rdf"),        tol: 5e-2) && allPass
allPass = report("pair_extra", d.pairExtra, flat(expected, "pair_extra"), tol: 5e-3) && allPass
allPass = report("frac",       d.frac,      flat(expected, "frac"),       tol: 1e-4) && allPass
allPass = report("glob",       d.glob,      flat(expected, "glob"),       tol: 1e-3) && allPass

// [2] env structural check (sampled; not a hard pass). For the rocksalt crystal
// every atom of a type is equivalent, so the Metal env_d rows should match the
// golden sample despite the different RNG.
if let env = json["env_seed0"] as? [String: Any] {
    let refD = flat(env, "env_d")
    let refT = flat(env, "env_t")
    var maxAbs: Float = 0
    for i in 0..<min(d.envD.count, refD.count) { maxAbs = max(maxAbs, abs(d.envD[i] - refD[i])) }
    var tMismatch = 0
    for i in 0..<min(d.envT.count, refT.count) where d.envT[i] != refT[i] { tMismatch += 1 }
    print("\n[2] env sets (structural)")
    print("  env_d max|Δ| = \(maxAbs)   env_t mismatches = \(tMismatch)/\(refT.count)")
}

// [3] decode + conformal parity on golden.log_probs
func parse2D(_ v: Any?) -> [[Double]] {
    guard let a = v as? [Any] else { return [] }
    return a.map { row in (row as? [Any])?.map { ($0 as? NSNumber)?.doubleValue ?? 0 } ?? [] }
}
let decodeGoldenPath = URL(fileURLWithPath: goldenPath)
    .deletingLastPathComponent().appendingPathComponent("decode_golden.json")
if let dg = try? Data(contentsOf: decodeGoldenPath),
   let dgj = try? JSONSerialization.jsonObject(with: dg) as? [String: Any] {
    let logp = parse2D(json["log_probs"])
    let frac = flat(expected, "frac").map { Double($0) }
    let prior = CompositionPrior()
    let marg = prior.marginals(logp, frac)
    let refMarg = parse2D(dgj["marginals"])
    var mMax = 0.0
    for t in 0..<min(marg.count, refMarg.count) {
        for c in 0..<min(marg[t].count, refMarg[t].count) {
            mMax = max(mMax, abs(marg[t][c] - refMarg[t][c]))
        }
    }
    let calib = DecodeAssets.calib
    var confOK = true
    let refConf = (dgj["conformal"] as? [Any])?.map { ($0 as? [Any])?.map { ($0 as? NSNumber)?.intValue ?? -1 } ?? [] } ?? []
    for t in 0..<marg.count {
        let got = Conformal.set(marg[t], calib).sorted()
        let ref = t < refConf.count ? refConf[t] : []
        if got != ref { confOK = false }
    }
    print("\n[3] decode + conformal")
    let mOK = mMax <= 1e-6
    print("\(mOK ? "✓" : "✗") marginals: max|Δ| = \(mMax)  (tol 1e-6)")
    print("\(confOK ? "✓" : "✗") conformal sets match")
    allPass = allPass && mOK && confOK
}

// --dump: write Metal descriptors for the ONNX bridge (onnx_bridge.py)
if let dp = dumpPath {
    let out: [String: Any] = [
        "nTypes": d.nTypes,
        "rdf": d.rdf.map { Double($0) },
        "pair_extra": d.pairExtra.map { Double($0) },
        "frac": d.frac.map { Double($0) },
        "glob": d.glob.map { Double($0) },
        "env_d": d.envD.map { Double($0) },
        "env_t": d.envT.map { Double($0) },
    ]
    let data = try JSONSerialization.data(withJSONObject: out)
    try data.write(to: URL(fileURLWithPath: dp))
    print("\ndumped Metal descriptors -> \(dp)")
}

// [4] optional CoreML model top-1
if let mp = modelPath {
    let runner = try CoreMLRunner(modelURL: URL(fileURLWithPath: mp))
    let logp = try runner.logProbs(d)
    let ref = json["top1_per_type"] as! [String]
    print("\n[4] model top-1 (CoreML)")
    for (t, row) in logp.enumerated() {
        let z = row.enumerated().max { $0.element < $1.element }!.offset + 1
        let sym = Elements.symbol(z)
        print("\(sym == ref[t] ? "✓" : "✗") type \(t): \(sym) (ref \(ref[t]))")
    }
}

print("\n" + (allPass ? "PASS" : "FAIL"))
exit(allPass ? 0 : 1)
