# Wildfires in Brazil

This project covers the exploration and modeling of wildfire records in Brazil from 2015 to 2024, collected from [INPE's Queimadas program](https://terrabrasilis.dpi.inpe.br/queimadas/portal/pages/secao_downloads/dados-abertos/#da-focos). This repository includes large parquet datasets versioned with DVC for secure sharing using a Google Cloud Platform (GCP) bucket as remote storage.

## Main structure
- `data/raw/queimadas_2015-2024.parquet`: the raw dataset manually aggregated from INPE's Queimadas platform.
- `data/processed/*.parquet`: aggregations ready for regression and classification tasks, as well as monthly and climatological summaries.
- `notebooks/0.1.0-eda-e-tratamento.ipynb` e `notebooks/1.0.0-eda-e-tratamento.ipynb`: versioned EDA and treatment notebooks.
- `scripts/`: helper scripts for data processing.
- `docs/`: project docs in Markdown format.
- `figures/`: static images generated during the analysis.

## Requirements
- Python 3.12+ 
- [uv](https://docs.astral.sh/uv/) to manage virtual environments and dependencies.
- DVC 3.x for data versioning.
- Dependencies listed in `pyproject.toml`.

## Installation
Create the virtual environment:
```powershell
uv venv
```

Install the dependencies:
```powershell
uv sync
```

## Download the datasets
The datasets are kept out of Git version control and are managed by DVC. After configuring the remote (`dvc remote list`), sync the `data/` folder:
```powershell
dvc pull data.dvc
```

## How to run the analysis
1. Activate the virtual environment.
2. Open the notebook `notebooks/1.0.0-eda-e-tratamento.ipynb`, update the dataset path if necessary and execute the cells to reproduce the EDA and treatment pipeline.

To get a better understanding of the data and the features engineered, consult `docs/descobertas.md` and `docs/features-dict.md`.
