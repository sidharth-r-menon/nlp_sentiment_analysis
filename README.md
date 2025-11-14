# Generalization of Transformer Models for Cross-Domain Sentiment Analysis


## Project Structure

```
nlp_sentiment_analysis/
├── data/
│   ├── raw/              # Raw datasets (IMDB, Amazon, Yelp)
│   │   ├── amazon/
│   │   │   ├── train.ft.txt
│   │   │   └── test.ft.txt
│   │   ├── imdb/
│   │   │   └── IMDB Dataset.csv
│   │   └── yelp/
│   │       ├── train.csv
│   │       ├── test.csv
│   │       └── readme.txt
│   └── processed/        # Preprocessed data (generated after running preprocess_data.py)
├── models/
│   ├── checkpoints/      # Model checkpoints during training
│   │   ├── bert_amazon/
│   │   │   └── best_model.pt
│   │   ├── bert_imdb/
│   │   │   └── best_model.pt
│   │   ├── bert_yelp/
│   │   │   └── best_model.pt
│   │   ├── distilbert_amazon/
│   │   │   └── best_model.pt
│   │   ├── distilbert_imdb/
│   │   │   └── best_model.pt
│   │   ├── distilbert_yelp/
│   │   │   └── best_model.pt
│   │   ├── roberta_amazon/
│   │   │   └── best_model.pt
│   │   ├── roberta_imdb/
│   │   │   └── best_model.pt
│   │   └── roberta_yelp/
│   │       └── best_model.pt
│   └── saved_models/     # Final trained model files
│       ├── bert_amazon.pt
│       ├── bert_imdb.pt
│       ├── bert_yelp.pt
│       ├── distilbert_amazon.pt
│       ├── distilbert_imdb.pt
│       ├── distilbert_yelp.pt
│       ├── roberta_amazon.pt
│       ├── roberta_imdb.pt
│       └── roberta_yelp.pt
├── metrics/              # Evaluation results (CSV files)
│   ├── baseline_results.csv
│   ├── bert_results.csv
│   ├── distilbert_results.csv
│   ├── roberta_results.csv
│   ├── bert_ensemble_results.csv
│   ├── distilbert_ensemble_results.csv
│   ├── roberta_ensemble_results.csv
│   └── combined_evaluation_results.csv
├── src/
│   ├── data/             # Data preprocessing
│   ├── train/            # Training scripts
│   ├── evaluation/       # Model evaluation
│   └── visualisation/    # Result visualization
│       └── figures/      # Generated plots and visualizations
├── app/                  # Streamlit demo application
│   └── app.py
├── assets/               # Styling assets
│   └── styles.css
├── requirements.txt
└── README.md
```

> **Note**: Due to file size constraints, the following directories are **not included** in this GitHub repository:
> - `data/raw/` - Raw datasets (download separately)
> - `data/processed/` - Generated after running preprocessing
> - `models/checkpoints/` - Training checkpoints (generated during training)
> - `models/saved_models/` - Trained model files (generated after training)

> The directory structure above shows how these folders will be organized once you run the training and evaluation scripts.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare the data:
```bash
python src/data/preprocess_data.py
```

## Training Models

### Train Baseline Models
```bash
python src/train/train_baseline.py
```

### Train Transformer Models
```bash
python src/train/train_transformer.py
```

### Train Ensemble Models
```bash
python src/train/train_ensemble.py
```

## Evaluation

Evaluate all models and generate metrics:
```bash
python src/evaluation/evaluator_baseline_transformer.py
python src/evaluation/evaluator_ensemble.py
```

## Visualization

Generate result plots:
```bash
cd src/visualisation
python visualise.py
```

Figures are saved to `src/visualisation/figures/`

## Demo Application

Run the Streamlit demo:
```bash
streamlit run app/app.py
```

## Models

- **Baseline**: TF-IDF + Logistic Regression
- **Transformers**: BERT, RoBERTa, DistilBERT
- **Ensemble**: Stacking ensemble with meta-classifier

## Datasets

### Amazon Product Reviews
- **Training samples**: 3,600,000
- **Test samples**: 400,000
- **Total**: 4,000,000 reviews
- **Split**: 72% train (~2,880,000), 8% validation (~320,000), 20% test (~800,000)
- **Classes**: Binary (positive/negative)
- **Source**: [Kaggle - Amazon Reviews](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)

### IMDB Movie Reviews
- **Total samples**: 50,000 reviews
- **Split**: 72% train (~36,000), 8% validation (~4,000), 20% test (~10,000)
- **Classes**: Binary (positive/negative)
- **Source**: [Kaggle - IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

### Yelp Business Reviews
- **Training samples**: 560,000
- **Test samples**: 38,000
- **Total**: 598,000 reviews
- **Split**: 72% train (~430,560), 8% validation (~47,840), 20% test (38,000)
- **Classes**: Binary (1-2 stars = negative, 3-4 stars = positive)
- **Source**: [Kaggle - Yelp Review Polarity](https://www.kaggle.com/datasets/irustandi/yelp-review-polarity)

> **Data Splitting**: All datasets are split into train (72%), validation (8%), and test (20%) sets during preprocessing. The validation set is primarily used for training the ensemble meta-classifier.
