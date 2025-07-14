# TopoClassifier

A ML-based repo for classifying hard-scatter from mixed (hard-scatter + pile-up) topoclusters used in ATLAS to reconstruct jets. It needs `.root`-files as input and processes them to `hdf5`-files and `.pt`-files for DNN and GNN (GCN and GAT) training.

## Quick Start
1. Clone the repo:  
   ```git clone https://github.com/Bastian125/TopoClassifier.git```  
2. Install dependencies found in ```dependencies.txt```  
3. Run any script; for help use the `-h` flag:  
   `python train.py -h`  

## Scripts Overview
- `config.py` – define plot settings, features to load from `.root`-files and paths for input and output files  
- `dataloader.py` – load and batch data  
- `preprocessing.py` – clean/prepare datasets  
- `models.py` – defines model architectures  
- `train.py` – train models  
- `evaluate.py` – evaluate performance of ML models  
- `plot.py` – visualise results  
- `check_graphs.py` – verify graph data integrity  
- `io_utils.py` – helper utilities

## Preprocessing
The preprocessing must be run first in order to generate the `.pt` and `.h5` files. Without this step the other scripts will not run.
```
Usage: preprocessing.py [-h] (--campaign {mc20a,mc20d,mc20e,mc23a,mc23d,mc23e,mc20,mc23} | --full | --print_features) [--no_normalisation] [--prepare_graphs] [--build_graphs]

Preprocess ROOT files and HDF5 splits.

Options:
  -h, --help
      Show this help message and exit.

  --campaign {mc20a,mc20d,mc20e,mc23a,mc23d,mc23e,mc20,mc23}
      Specify the campaign for preprocessing or renormalisation.

  --full
      Run full preprocessing on all datasets.

  --print_features
      Print all features in the root file.

  --no_normalisation
      Skip normalisation and time transformation.

  --prepare_graphs
      Prepare HDF5 files for graph building.

  --build_graphs
      Build and save PyG graphs from normalised HDF5 splits.

```

## Plotting
Creates plots for physics analysis.
```
Usage: plot.py [-h] (--avgMu | --NPV | --run_comparison | --NPV_comparison | --high_response | --response | --response_noPU_vs_PU | --PU_response | --all)

Plot cluster features for MC20a/d/e and MC23a/d/e.

options:
  -h, --help            show this help message and exit
  --avgMu               Plots distribution of avgMu for both campaigns
  --NPV                 Plots distribution of n_PV for both campaigns
  --run_comparison      Plot comparison of every feature for Run 2 and Run 3.
  --NPV_comparison      Plot every feature for different n_PV bins for both campaigns.
  --high_response       Plot every feature of both campaigns for cluster response lower or equal to 40 or higher than 40 for comparison.
  --response            Creates response plots for different n_PV bins for both campaigns.
  --response_noPU_vs_PU
                        Creates response plots for different n_PV bins, and no pile-up for both campaigns.
  --PU_response         Plot mean and median cluster response in n_PV bins for clusters with the complete energy range,clusters with energy lower than 100~GeV, and clusters with energy greater than or equal to 100~GeV
  --all                 Make every plot.

```
