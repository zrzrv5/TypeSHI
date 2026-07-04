// Descriptor constants — MUST stay in lockstep with src/typeid2elem/descriptors.py.
// If you change a value here, change it there (and vice-versa) or parity breaks.
import Foundation

public enum Desc {
    public static let rMax: Float = 8.0          // neighbor cutoff (Angstrom)
    public static let nBins = 64                 // partial-RDF bins
    public static let dr: Float = rMax / Float(nBins)   // 0.125 A
    public static let smearBins: Float = 1.0     // Gaussian sigma (in bins) on the raw histogram
    public static let cnRadii: [Float] = [2, 3, 4, 6]   // running-coordination checkpoints
    // pair_extra layout: [log1p(cn) x4, nn_median/rMax, nn_p10/rMax, peak_pos, log1p(peak_h)]
    public static let nPairExtra = cnRadii.count + 4     // 8
    public static let nGlob = 3                  // [log number density, has_cell, 1/nTypes]

    // learned environment sets (v3): per type, M sampled atoms x K nearest neighbors <= rEnv,
    // stored as (distance, partner type id); -1 pads. Keeps per-atom JOINT coordination.
    public static let mEnv = 16
    public static let kEnv = 16
    public static let rEnv: Float = 6.0

    public static let maxTypes = 8               // exported CoreML fixed type dim (T_FIXED)
    public static let nClasses = 94              // atomic numbers 1..94, class = Z-1
}
