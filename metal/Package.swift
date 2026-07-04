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
    ],
    targets: [
        .target(
            name: "InaccurateRDFCalculator",
            // .metal files in the target are compiled into a default.metallib
            // and loaded via device.makeDefaultLibrary(bundle: .module).
            resources: [.process("Shaders.metal")]
        ),
        .executableTarget(
            name: "ParityCheck",
            dependencies: ["InaccurateRDFCalculator"],
            // golden.json is read at runtime relative to the repo, not bundled.
            path: "Sources/ParityCheck"
        ),
    ]
)
