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
// Split of labor:
//   rdfHistogram — the two heavy O(N * neighbors) aggregates:
//     (1) per-(typeA,typeB) distance histogram  -> partial RDF after CPU normalize
//     (2) per-center nearest distance to each type -> robust NN median/p10 on CPU
//   envSample    — per sampled center, its K nearest neighbors (distance, type)
//     within R_ENV, sorted by distance (the learned-encoder env sets).
// Everything else (normalize, smear, CN, peak, frac, glob) is T x T sized and
// cheap; it lives in Swift.
//
// Periodicity: matscipy's neighbour_list enumerates ALL periodic images (and an
// atom's own images) within the cutoff — NOT just the single nearest image. For
// the small cells common here (the parity cell is 8.4 A with rMax 8.0, so the
// half-cell is well below the cutoff) minimum-image is wrong. We reproduce
// matscipy by looping over image replicas s in [-n,n]^3 per axis, where n_axis =
// ceil(rMax / perpendicular_height_axis) is computed on the host (0 for an open
// box). Exact for triclinic cells too: the shift is s.x*a0 + s.y*a1 + s.z*a2.

struct Params {
    float rMax;
    float dr;
    float rEnv;
    uint  nBins;
    uint  nTypes;
    uint  nAtoms;
    int   nx;      // image replicas per axis (0 for open box / no images)
    int   ny;
    int   nz;
    uint  pbc;     // 1 = periodic, 0 = open box
};

// One thread per CENTER atom i. Brute-force over all j and all periodic image
// shifts within the cutoff. (Replace the j-loop with a uniform-grid / cell-list
// traversal for large N — the interior-crop trick from descriptors.py maps
// directly to grid cells; correctness first, that stays a perf TODO.)
//
// hist:   [nTypes * nTypes * nBins] uint, ordered (a<-b) = a*nTypes*nBins + b*nBins + bin
// nnDist: [nAtoms * nTypes] float bits, min distance from atom i to any atom of
//         type b (initialized to rMax on the host), kept with a real atomic-min.
kernel void rdfHistogram(
    device const float3* pos       [[buffer(0)]],
    device const uint*   types     [[buffer(1)]],
    device       atomic_uint* hist [[buffer(2)]],
    device       atomic_uint* nnDist [[buffer(3)]],   // float bits, atomic-min
    constant     Params& p         [[buffer(4)]],
    device const float3* cell      [[buffer(5)]],      // 3 lattice vectors (rows)
    uint i [[thread_position_in_grid]])
{
    if (i >= p.nAtoms) return;
    const uint ta = types[i];
    const float3 pi = pos[i];
    const uint nb = p.nBins;
    const float rMax = p.rMax;

    for (uint j = 0; j < p.nAtoms; ++j) {
        const float3 pj = pos[j];
        const uint tb = types[j];
        for (int sx = -p.nx; sx <= p.nx; ++sx)
        for (int sy = -p.ny; sy <= p.ny; ++sy)
        for (int sz = -p.nz; sz <= p.nz; ++sz) {
            if (j == i && sx == 0 && sy == 0 && sz == 0) continue;  // skip self image at 0
            float3 shift = float(sx) * cell[0] + float(sy) * cell[1] + float(sz) * cell[2];
            float r = length(pj + shift - pi);
            if (r >= rMax) continue;

            uint bin = min(uint(r / p.dr), nb - 1);
            uint idx = (ta * p.nTypes + tb) * nb + bin;
            atomic_fetch_add_explicit(&hist[idx], 1u, memory_order_relaxed);

            // per-center nearest distance to type tb: atomic-min on the float bit
            // pattern. For non-negative floats the IEEE-754 bits are monotonic, so
            // an unsigned atomic_min on the bits is a float min.
            uint ni = i * p.nTypes + tb;
            atomic_fetch_min_explicit(&nnDist[ni], as_type<uint>(r), memory_order_relaxed);
        }
    }
}

// Per SAMPLED center, gather its K_ENV nearest neighbors within R_ENV, ordered by
// distance, as (distance, partner type). Mirrors descriptors._compute_env: the
// host picks up to M_ENV atoms per type (seeded, without replacement) and lays
// them out as flat "chosen" arrays; one thread handles one chosen atom.
//
//   chosenAtom[c] = atom index i        chosenType[c] = its type a
//   chosenSlot[c] = its slot m in [0,M) envD/envT written at (a*M + m)*K + k
//
// env is a *sampled* feature (high variance by design); the Python path pools
// several draws at inference. TypeSHI.predict runs this with several seeds and
// averages the model log-probs.
kernel void envSample(
    device const float3* pos        [[buffer(0)]],
    device const uint*   types      [[buffer(1)]],
    device const uint*   chosenAtom [[buffer(2)]],
    device const uint*   chosenType [[buffer(3)]],
    device const uint*   chosenSlot [[buffer(4)]],
    constant     Params& p          [[buffer(5)]],
    device const float3* cell       [[buffer(6)]],
    constant     uint&   nChosen    [[buffer(7)]],
    constant     uint&   mEnv       [[buffer(8)]],
    constant     uint&   kEnv       [[buffer(9)]],
    device       float*  envD       [[buffer(10)]],
    device       float*  envT       [[buffer(11)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= nChosen) return;
    const uint i = chosenAtom[c];
    const uint a = chosenType[c];
    const uint m = chosenSlot[c];
    const float3 pi = pos[i];
    // The thread-local top-K buffer is fixed at 16; it MUST be >= Constants.kEnv
    // (Desc.kEnv == K_ENV == 16). K clamps to the buffer so bestD/bestT are never
    // indexed out of range even if kEnv were mis-set larger than the buffer.
    const uint K = min(kEnv, 16u);
    const float rEnv = p.rEnv;
    float bestD[16];
    int   bestT[16];
    uint  cnt = 0;
    for (uint t = 0; t < 16u; ++t) { bestD[t] = 0.0; bestT[t] = -1; }

    for (uint j = 0; j < p.nAtoms; ++j) {
        const float3 pj = pos[j];
        const int    tb = int(types[j]);
        for (int sx = -p.nx; sx <= p.nx; ++sx)
        for (int sy = -p.ny; sy <= p.ny; ++sy)
        for (int sz = -p.nz; sz <= p.nz; ++sz) {
            if (j == i && sx == 0 && sy == 0 && sz == 0) continue;
            float3 shift = float(sx) * cell[0] + float(sy) * cell[1] + float(sz) * cell[2];
            float r = length(pj + shift - pi);
            if (r > rEnv) continue;
            if (cnt == K && r >= bestD[K - 1]) continue;   // full and not smaller
            uint hi = (cnt < K) ? cnt : (K - 1);           // last writable slot
            int ins = int(hi);
            while (ins > 0 && bestD[ins - 1] > r) {
                bestD[ins] = bestD[ins - 1];
                bestT[ins] = bestT[ins - 1];
                --ins;
            }
            bestD[ins] = r;
            bestT[ins] = tb;
            if (cnt < K) ++cnt;
        }
    }

    const uint outBase = (a * mEnv + m) * kEnv;
    for (uint t = 0; t < kEnv; ++t) {
        if (t < K) {                       // guard (not ?:) so bestD[t] is only read when in-range
            envD[outBase + t] = bestD[t];
            envT[outBase + t] = float(bestT[t]);
        } else {
            envD[outBase + t] = 0.0;       // pad distance
            envT[outBase + t] = -1.0;      // pad type
        }
    }
}
