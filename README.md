# Workflow-CI — Dicoding MSML Proyek Akhir (Kriteria 3)

Titanic survival prediction, re-training via MLflow Project + GitHub Actions CI.

## Struktur Repository

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── main.yml
├── MLProject/
│   ├── MLproject
│   ├── conda.yaml
│   ├── modelling.py
│   ├── requirements.txt
│   └── titanic_preprocessing/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── preprocessing/
│   ├── titanic_raw.csv
│   └── preprocessing_titanic.py
├── README.md
└── .gitignore
```

## Cara Menjalankan

1. Install dependencies:
   ```bash
   pip install -r MLProject/requirements.txt
   ```

2. Jalankan MLflow Project:
   ```bash
   mlflow run ./MLProject --entry-point main --env-manager local
   ```

## CI / GitHub Actions

Pada setiap push ke branch `main` (atau melalui `workflow_dispatch`), GitHub Actions akan:

- Menjalankan ulang pelatihan model RandomForest
- Menyimpan artefak model (`model.pkl`) kembali ke repository
- Melakukan commit otomatis untuk memperbarui artefak tersebut

Ini memenuhi persyaratan **Skilled** (Kriteria 3) yaitu artefak model dipersist ke dalam repository.

## Environment

- Python 3.12.7
- mlflow==2.19.0
- scikit-learn, pandas, numpy, joblib
