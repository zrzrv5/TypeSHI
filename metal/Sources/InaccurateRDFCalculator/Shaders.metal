#include <metal_stdlib>
using namespace metal;

// InaccurateRDFCalculator GPU front-end.
//
// "Inaccurate" is deliberate: the on-device descriptor may subsample RDF centers
// and use a coarse neighbor search. Chemistry is decided by bond-length *peaks*
// and coordination, both robust to dropping a fraction of centers — so an
// approximate g_ab(r) is allowed as long as top-1 per type and the 90% conformal
// set match the reference (see metal/parity/golden.json).
//
// Split of labor: this kernel emits the two heavy, O(N * neighbors) aggregates —
//   (1) per-(typeA,typeB) distance histogram   -> partial RDF after CPU normalize
//   (2) per-center nearest distance to each type -> robust NN median/p10 on CPU
// Everything downstream (normalize, smear, CN, peak, frac, glob, env) is T x T
// sized and cheap; do it in Swift.

struct Params {
    float  rMax;
    float  dr;
    uint   nBins;
    uint   nTypes;
    uint   nAtoms;
    float3 cell0;      // cell rows (Angstrom); for minimum-image PBC
    float3 cell1;
    float3 cell2;
    uint   pbc;        // 1 = periodic (minimum image), 0 = open box
};

// Minimum-image displacement under a (possibly triclinic) cell.
// TODO(perf): precompute the inverse cell on the host and pass it in; this
// assumes a near-orthorhombic cell (good enough for the first cut).
static inline float3 minImage(float3 d, constant Params& p) {
    if (p.pbc == 0) return d;
    float3 L = float3(p.cell0.x, p.cell1.y, p.cell2.z);   // orthorhombic lengths
    d -= L * rint(d / L);
    return d;
}

// One thread per CENTER atom i. Brute-force over all j (the simple first version;
// replace the inner loop with a uniform-grid / cell-list traversal for large N —
// the interior-crop trick from descriptors.py maps directly to grid cells).
//
// hist:   [nTypes * nTypes * nBins] uint, ordered (a<-b) = a*nTypes*nBins + b*nBins + bin
// nnDist: [nAtoms * nTypes] float, min distance from atom i to any atom of type b
//         (initialized to rMax on the host)
kernel void rdfHistogram(
    device const float3* pos      [[buffer(0)]],
    device const uint*   types    [[buffer(1)]],
    device       atomic_uint* hist[[buffer(2)]],
    device       float*  nnDist   [[buffer(3)]],   // atomic-min emulated below
    constant     Params& p        [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= p.nAtoms) return;
    const uint ta = types[i];
    const float3 pi = pos[i];
    const uint nb = p.nBins;

    for (uint j = 0; j < p.nAtoms; ++j) {
        if (j == i) continue;
        float3 d = minImage(pos[j] - pi, p);
        float r = length(d);
        if (r >= p.rMax) continue;
        const uint tb = types[j];

        uint bin = min(uint(r / p.dr), nb - 1);
        uint idx = (ta * p.nTypes + tb) * nb + bin;
        atomic_fetch_add_explicit(&hist[idx], 1u, memory_order_relaxed);

        // per-center nearest distance to type tb (CPU aggregates median/p10)
        uint ni = i * p.nTypes + tb;
        // NOTE: races on nnDist are benign for a *min*, but do it right:
        // TODO replace with atomic_min on a uint bitcast, or a second reduction
        // kernel. For the skeleton, a relaxed compare-store is close enough to
        // validate the pipeline; fix before trusting nn_p10.
        if (r < nnDist[ni]) nnDist[ni] = r;
    }
}

// TODO(kernel): envSample — per type, pick M atoms, gather their K nearest
// neighbors (distance, partner type) sorted by distance. Needs a per-type atom
// list (prefix-sum bucketing on the host) and a small per-thread top-K heap.
// Env is a *sampled* feature (high variance by design); the Python path pools 4
// draws at inference. Mirror that: run this kernel 4x with different seeds and
// average log-probs on the CoreML side.
