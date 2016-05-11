# Standardized MSM pipeline
We have a large amount of simulation data from many related systems. Instead of selecting a different model-building approach for each system, we would like to select and automate a uniform MSM-building procedure, so that we can more easily compare MSMs.

## Pipeline outline in one line
Trajectories --> discrete trajectories --> statistics

## Details
1. **Inputs**:
   - List of filenames of munged hdf5 trajectories
      - These are the outputs of the FAHmunge pipeline
   - Lag-time between recorded snapshots
   - Lag-time for MSM estimation
   - Metastability lag-time
2. **Indexing**:
   - Build an index of `filename : nframes` pairs.
   - Note that these files can be written to-- `nframes` can increase over time-- can't assume fixed file length, but can assume that the only file modifications will be to append to the end of the trajectory.
3. **Discretizing** w.r.t. an appropriate metric
   - Two typical choices of metric:
      - Explicit-feature-based kinetic distance:
         - Compute *features* (angles, inter-residue distances, etc.)
            - Save computed features as a `.npz` archive
         - Use tICA to find a linear projection where Euclidean distance approximates diffusion distance
      - MinRMSD
   - Number of clusters. For now, we will fix this to be a large-ish number (1000), as we can recover from an overly fine discretization by "lumping" later.
   - Choice of clustering algorithm: for now, fix to minibatch k-means.
   - Save discrete trajectories as a `.npz` archive
4. **Statistics** of discretized trajectories
   - Distribution of trajectory lengths
   - "Ergodic trimming" -- select the largest ergodic subspace, and estimate a Markov model
   - Test Markovianity
      - Implied timescales plots
      - Chapman-Kolmogorov tests among macrostates (macrostates will be defined below!)
   - "Lump" microstates into macrostates
      - Using eigenvalues of transition matrix

## Major remaining challenges
1. What do we do for "disconnected" models?
   - In cases where the observed dynamics are not ergodic, what should we do?
2. What scheme gives the best model?
