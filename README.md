## TDT05 group project
This is a project that explores the performance increase from using a backbone trained using a self-supervised method (SimSiam) before training in a more traditional supervised way.

## Development

### Installing dependecies
You should use a virtual environment. Thus do the following steps for the first time setup:
```bash
python -m venv venv
source venv/bin/activate  # windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Datasets:

#### Supervised
Cat breeds image dataset: https://www.kaggle.com/datasets/shawngano/gano-cat-breed-image-collection 

#### Self-Supervised 
Animals10: https://www.kaggle.com/datasets/alessiocorrado99/animals10

## Training:
#### Self-supervised:
1. Update the parameters in `train_ssl.py` to your liking.
2. Run `python train_ssl.py`

#### Supervised:
1. Update the parameters in `train_supervised.py` to your liking.
2. Run `python train_supervised.py`
