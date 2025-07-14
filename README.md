# TopoClassifier

A ML-based repo for classifying hard-scatter from mixed (hard-scatter + pile-up) topoclusters used in ATLAS to reconstruct jets. It needs `.root`-files as input and processes them to `hdf5`-files and `.pt`-files for DNN and GNN (GCN and GAT) training.

## Quick Start
1. Clone the repo:  
   ```git clone https://github.com/Bastian125/TopoClassifier.git```  
2. Install dependencies found in ```dependencies.txt```  
3. Run any script; for help use the `-h` flag:  
   `python train.py -h`  

## Scripts Overview
- `config.py` – define plot settings and features to load from `.root`-files  
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
