import Foundation
import InaccurateRDFCalculator
import simd

// Parity harness: reproduce metal/parity/golden.json's descriptors with the
// Metal path and diff against the Python reference.
//
//   swift run parity-check [path/to/golden.json] [path/to/model.mlpackage]
//
// The descriptor comparison (rdf/pair_extra/frac/glob) needs no model. Pass the
// mlpackage to also check top-1 per type against the reference.

let args = CommandLine.arguments
let goldenPath = args.count > 1 ? args[1] : "metal/parity/golden.json"
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
func flat(_ key: String) -> [Float] {
    func rec(_ v: Any) -> [Float] {
        if let d = v as? Double { return [Float(d)] }
        if let i = v as? Int { return [Float(i)] }
        if let a = v as? [Any] { return a.flatMap(rec) }
        return []
    }
    return rec(expected[key]!)
}

let comp = try DescriptorComputer()
let d = comp.compute(structure)

func report(_ name: String, _ got: [Float], _ ref: [Float], tol: Float) {
    guard got.count == ref.count else {
        print("✗ \(name): shape \(got.count) vs \(ref.count)"); return
    }
    var maxAbs: Float = 0
    for i in 0..<got.count { maxAbs = max(maxAbs, abs(got[i] - ref[i])) }
    print("\(maxAbs <= tol ? "✓" : "✗") \(name): max|Δ| = \(maxAbs)  (tol \(tol))")
}

print("structure: \(positions.count) atoms, \(d.nTypes) types\n")
report("rdf",        d.rdf,       flat("rdf"),        tol: 5e-2)
report("pair_extra", d.pairExtra, flat("pair_extra"), tol: 5e-3)
report("frac",       d.frac,      flat("frac"),       tol: 1e-4)
report("glob",       d.glob,      flat("glob"),       tol: 1e-3)

if args.count > 2 {
    let runner = try CoreMLRunner(modelURL: URL(fileURLWithPath: args[2]))
    let logp = try runner.logProbs(d)
    let ref = json["top1_per_type"] as! [String]
    for (t, row) in logp.enumerated() {
        let z = row.enumerated().max { $0.element < $1.element }!.offset + 1
        let sym = Elements.symbol(z)
        print("\(sym == ref[t] ? "✓" : "✗") type \(t): \(sym) (ref \(ref[t]))")
    }
}
