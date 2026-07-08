// swift-tools-version: 5.9
// InaccurateRDFCalculator — on-device (Metal + CoreML) inference for TypeSHI.
//
// Status: SKELETON. The descriptor front-end (Metal) and the CoreML runner have
// real API surfaces and host wiring; the GPU kernel bodies and env sampling are
// marked TODO. Build + iterate on a Mac; parity target is metal/parity/golden.json.
import PackageDescription

let package = Package(
    name: "InaccurateRDFCalculator",
    platforms: [.macOS(.v14), .iOS(.v16)],
    products: [
        .library(name: "InaccurateRDFCalculator",
                 targets: ["InaccurateRDFCalculator"]),
        .executable(name: "parity-check", targets: ["ParityCheck"]),
        .executable(name: "desc-dump", targets: ["DescDump"]),
    ],
    targets: [
        .target(
            name: "InaccurateRDFCalculator",
            // Shaders.metal is bundled as a resource and compiled at runtime via
            // device.makeLibrary(source:) — SwiftPM copies (does not compile) it,
            // and runtime source-compilation is portable across macOS + iOS.
            // decode_assets.json holds the baked PMI + conformal calibration.
            resources: [
                .process("Shaders.metal"),
                .process("Resources/decode_assets.json"),
            ]
        ),
        .executableTarget(
            name: "ParityCheck",
            dependencies: ["InaccurateRDFCalculator"],
            // golden.json is read at runtime relative to the repo, not bundled.
            path: "Sources/ParityCheck"
        ),
        .executableTarget(
            name: "DescDump",
            dependencies: ["InaccurateRDFCalculator"],
            path: "Sources/DescDump"
        ),
    ]
)
