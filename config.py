# config.py

# --- Feature Columns --- #
columns = [
    "clusterE",
    "cluster_FIRST_ENG_DENS",
    "cluster_EM_PROBABILITY",
    "cluster_CENTER_LAMBDA",
    "cluster_CENTER_MAG",
    "cluster_nCells_tot",
    "cluster_ENG_FRAC_EM",
    "cluster_SECOND_TIME",
    "cluster_AVG_TILE_Q",
    "cluster_AVG_LAR_Q",
    "cluster_SECOND_R",
    "cluster_LATERAL",
    "cluster_time",
    "cluster_ISOLATION",
    "cluster_ENG_CALIB_TOT",
    "cluster_SIGNIFICANCE",
    "nPrimVtx",
    "avgMu",
]

log_features = [
    "clusterE",
    "cluster_FIRST_ENG_DENS",
    "cluster_CENTER_LAMBDA",
    "cluster_nCells_tot",
    "cluster_SECOND_TIME",
    "cluster_AVG_TILE_Q",
    "cluster_AVG_LAR_Q",
    "cluster_SECOND_R",
]

normal_features = [
    "clusterE",
    "cluster_FIRST_ENG_DENS",
    "cluster_EM_PROBABILITY",
    "cluster_CENTER_LAMBDA",
    "cluster_CENTER_MAG",
    "cluster_nCells_tot",
    "cluster_ENG_FRAC_EM",
    "cluster_SECOND_TIME",
    "cluster_AVG_TILE_Q",
    "cluster_AVG_LAR_Q",
    "cluster_SECOND_R",
    "cluster_LATERAL",
    "cluster_ISOLATION",
]

# --- Paths --- #
root_path = "/ceph/e4/users/cdelitzsch/public/forClusterStudies/"
save_path = "/ceph/e4/users/bschuchardt/public/MA/data/"
