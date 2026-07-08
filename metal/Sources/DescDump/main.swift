import Foundation
import InaccurateRDFCalculator
import simd

// Compute Metal descriptors for an arbitrary structure and write them as JSON.
// Lets the Python side verify the Metal descriptor path on real eval files (not
// just the synthetic golden) and push them through the deploy model.
//
//   swift run desc-dump <input.json> <out.json> [envSeed]
//
// input.json: {"positions": [[x,y,z],...], "type_ids": [int,...],
//              "cell": [[..],[..],[..]] | null}

let args = CommandLine.arguments
guard args.count >= 3 else {
    FileHandle.standardError.write("usage: desc-dump <input.json> <out.json> [envSeed]\n".data(using: .utf8)!)
    exit(2)
}
let inURL = URL(fileURLWithPath: args[1])
let outURL = URL(fileURLWithPath: args[2])
let seed = args.count > 3 ? UInt64(args[3]) ?? 0 : 0

let obj = try JSONSerialization.jsonObject(with: Data(contentsOf: inURL)) as! [String: Any]
let posArr = obj["positions"] as! [[Double]]
let typeArr = obj["type_ids"] as! [Int]
let positions = posArr.map { SIMD3<Float>(Float($0[0]), Float($0[1]), Float($0[2])) }
let typeIds = typeArr.map { UInt32($0) }
var cell: (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)? = nil
if let c = obj["cell"] as? [[Double]] {
    cell = (SIMD3<Float>(c[0].map(Float.init)),
            SIMD3<Float>(c[1].map(Float.init)),
            SIMD3<Float>(c[2].map(Float.init)))
}
let structure = Structure(positions: positions, typeIds: typeIds, cell: cell)

let comp = try DescriptorComputer()
let d = comp.compute(structure, envSeed: seed)
let out: [String: Any] = [
    "nTypes": d.nTypes,
    "rdf": d.rdf.map { Double($0) },
    "pair_extra": d.pairExtra.map { Double($0) },
    "frac": d.frac.map { Double($0) },
    "glob": d.glob.map { Double($0) },
    "env_d": d.envD.map { Double($0) },
    "env_t": d.envT.map { Double($0) },
]
try JSONSerialization.data(withJSONObject: out).write(to: outURL)
print("desc-dump: \(positions.count) atoms, \(d.nTypes) types -> \(outURL.lastPathComponent)")
