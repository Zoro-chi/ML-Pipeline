## Project: Data Pipeline with DVC and MLflow for Machine Learning

This project demonstrates how to build an end to end data pipeline using DVC for data and model versioning and MLflow for experiment tracking.
The pipeline focuses on training a RandomForest Classifier model on the Pima Indians Diabetes dataset, with clear stages for data processing, model training, and model evaluation.

### Add the preprocess stage to the DVC pipeline

```bash
dvc stage add -n preprocess \
 -p preprocess.input, preprocess.output \
 -d src/preprocess.py -d data/raw/data.csv \
 -o data/processed/data.csv \
 python src/preprocess.py
```

### Add the train stage to the DVC pipeline

```bash
dvc stage add -n train \
 -p train.data, train.model, train.random_state, train.n_estimators, train.max_depth \
 -d src/train.py -d data/raw/data.csv \
 -o models/model.pkl \
 python src/train.py
```

### Add the evaluate stage to the DVC pipeline

```bash
dvc stage add -n evaluate \
    -d src/evaluate.py -d models/model.pkl -d data/raw/data.csv \
    python src/evaluate.py
```
