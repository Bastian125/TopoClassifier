# TopoClassifier

A ML-based repo for classifying hard-scatter from mixed (hard-scatter + pile-up) topoclusters used in ATLAS to reconstruct jets.

## Quick Start
1. Clone the repo:  
   ```git clone https://github.com/Bastian125/TopoClassifier.git```  
2. Install dependencies found in ```dependencies.txt```  
3. Run any script; for help use the `-h` flag:  
   `python train.py -h`  

## Scripts Overview
- `dataloader.py` – load and batch data  
- `preprocessing.py` – clean/prepare datasets  
- `models.py` – defines model architectures  
- `train.py` – train models  
- `evaluate.py` – evaluate performance  
- `plot.py` – visualise results  
- `check_graphs.py` – verify graph data integrity  
- `io_utils.py`, `config.py`, `plot.py` – helper utilities
