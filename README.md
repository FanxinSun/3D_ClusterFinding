# 3D_ClusterFinding

Machine-learning side of the sPHENIX TPC cluster-finding project: notebooks for
hit labelling, dataset preparation and model prototyping on simulated TPC data.

Status (2026-10-07): groundwork plus a first real-data product. The notebooks
under GroundTruth_Making/ are the hand-labelling era; the Python scripts beside
them recover, for the real TPC ntuple, which hits belong to each official
cluster (certified by exact ADC-sum, max-ADC, bounding-box and centroid
identities; 0.88 of clusters over 100 events), attach the tracker's on-track
flags, compute 11 cluster descriptors and census the hot pads. Their outputs are
kept out of git (data/). Simulated training data come from the TPC_Sim_Pipeline
exports, interface version v6.2 (v6.3 in preparation); their layout is
documented in that repository's README. See MIGRATION.md for setting this
project up on a new machine.

It consumes datasets produced by the TPC_Sim_Pipeline repository, whose exporter
writes the training tables this side reads. Data files themselves are not
tracked here.

RUN_REF = the reference real-data run; identifier intentionally not recorded here.
