# ZoidBerg2.0 — Computer Aided Diagnosis (Pneumonia from Chest X-rays)

Classical ML pipeline that classifies chest X-rays as `NORMAL` vs `PNEUMONIA`,
using resize + flatten + PCA + scikit-learn estimators. Implements all three
evaluation strategies required by the project brief:

1. Simple train/test split.
2. Train / validation / test split with hyperparameter tuning.
3. Stratified K-fold cross-validation.

## Layout

```
ZoidBerg/
├── chest_Xray/chest_Xray/{train,val,test}/{NORMAL,PNEUMONIA}/   # Kaggle dataset
├── Jupyter/zoidberg2.ipynb                                       # main notebook
├── artifacts/                                                    # cached features + saved models
├── requirements.txt
└── README.md
```

## Setup (Windows / PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name zoidberg2 --display-name "Python (zoidberg2)"
jupyter notebook Jupyter/zoidberg2.ipynb
```

Then pick the kernel **Python (zoidberg2)** inside the notebook.

## Notes

- `chest_Xray/` is the dataset that ships with the project.
- Image features and trained models are cached under `artifacts/` so the notebook
  does not have to re-decode every JPEG on each run.
- Both `artifacts/` and `.venv/` are gitignored.
