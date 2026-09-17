# ML Training Strategy for Real TPC Datasets

## 1. The Core Challenge: Extreme Class Imbalance and "Needle in a Haystack" Tracking

When transitioning from simulated Monte Carlo (Geant4) data to real-world detector data, the complexity of track finding increases exponentially. The analysis of our real dataset (`clusters_seeds_island_RUN_REF-0.root_ntuplizer.root`) reveals a profound class imbalance between signal and background noise.

### 1.1 Hardware Hits vs. Reconstructed Tracks
In our raw data tree (`ntp_hit`), the variable `hitID` does **not** represent a full particle track (as it does in simulated data). Instead, it represents a localized "hardware island"—a cluster of adjacent readout pads that fired simultaneously. 

*   **Average Layer Span:** ~1.4 layers
*   **Conclusion:** `hitID` is merely a localized 3D space-point, not a trajectory.

### 1.2 The "Needle in a Haystack"
When we examine a specific collision (e.g., Event 1.0), the disparity between raw data and actual physics tracks is staggering:
*   **Total Active Hits:** ~58,339 individual sensor firings.
*   **Raw Islands (`hitID`s):** ~23,825 grouped clusters.
*   **Valid Classical Tracks (`seedID`s):** Only **26** successfully reconstructed tracks.

**The Implications:**
If 26 valid tracks span an average of 40 layers, they account for roughly 1,040 islands. This means that out of 23,825 islands, **~95% is pure background noise** (delta rays, electronic noise, beam background). 

Any Machine Learning model trained on this data must learn to ignore 95% of the input space to thread trajectories through the remaining 5%.

---

## 2. Proposed ML Training Solution: Pseudo-Ground Truth

Because real data lacks the absolute "Truth Track IDs" found in simulations, we must construct a **Pseudo-Ground Truth** to provide training targets for our models.

### 2.1 The Training Strategy
The classical tracking algorithm outputs a highly curated tree: `ntp_clus_trk`. This tree contains only the clusters that were successfully fitted into valid tracks, grouped by `seedID`. 

We will use this to create binary classification targets:

1.  **Do NOT use `ntp_hit`'s `hitID` as a target label.** It is only a spatial feature.
2.  **Define the Signal (Class 1):** Use the `ntp_clus_trk` tree. Clusters that are grouped into a `seedID` represent our positive "Tracked" class.
3.  **Define the Noise (Class 0):** Map the spatial coordinates (x, y, z) or the hardware indices from the curated `ntp_clus_trk` back onto the raw `ntp_hit` tree. Any raw hit that does *not* spatially correspond to a tracked cluster is labeled as "Untracked" (Noise).

### 2.2 Model Architecture Considerations
Given the extreme 95% noise ratio, standard loss functions (like basic Cross-Entropy) will heavily bias the model toward predicting "Noise" for everything.

*   **Loss Functions:** We must implement Focal Loss or heavily weighted Cross-Entropy to penalize the model for missing the rare "Signal" tracks.
*   **Data Representation:** Graph Neural Networks (GNNs) are highly recommended. By constructing graphs where nodes are raw islands, the GNN can learn edge-weights to connect the 5% signal nodes while actively dropping edges to the 95% noise nodes.
