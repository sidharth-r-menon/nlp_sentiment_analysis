import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

config = {
    "models": {
        "baseline": {
            "name": "tfidf_logreg",
            "tfidf_params": {
                "max_features": 10000,
                "ngram_range": (1, 2),
                "min_df": 5,
            },
            "logreg_params": {
                "C": 1.0,
                "max_iter": 1000,
                "class_weight": "balanced",
            },
        },
        "transformers": [
            {"name": "bert", "model_name": "bert-base-uncased", "num_labels": 2},
            {"name": "roberta", "model_name": "roberta-base", "num_labels": 2},
            {"name": "distilbert", "model_name": "distilbert-base-uncased", "num_labels": 2},
        ],
    },

    "training": {
        "batch_size": 64,
        "num_epochs": 3,
        "learning_rate": 2.0e-5,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "fp16": True,
        "save_steps": 1000,
        "eval_steps": 500,
        "logging_steps": 100,
        "save_total_limit": 2,
        "dataloader_num_workers": 4,
        "pin_memory": True,
    },

    "cross_domain": {
        "train_test_pairs": [
            {"train": "imdb", "test": "amazon"},
            {"train": "imdb", "test": "yelp"},
            {"train": "amazon", "test": "imdb"},
            {"train": "amazon", "test": "yelp"},
            {"train": "yelp", "test": "imdb"},
            {"train": "yelp", "test": "amazon"},
        ]
    },

    "evaluation": {
        "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"]
    },

    "visualization": {
        "confusion_matrix": True,
        "roc_curve": True,
        "attention_heatmap": True,
        "shap_analysis": True,
        "top_k_tokens": 10,
    },
}

data = None

with open(r'..\..\data\processed\preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

domains = list(data.keys())
for train_domain in domains:
    train_df = data[train_domain]['train']
    val_df = data[train_domain]['val']

    train_data = train_df['text'].tolist()
    train_labels = train_df['label'].tolist()
    val_data = val_df['text'].tolist()
    val_labels = val_df['label'].tolist()

    print("Training")

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2),min_df=5)
    train_X = vectorizer.fit_transform(train_data)

    classifier = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
    classifier.fit(train_X, train_labels)

    os.makedirs(f'..\..\models\saved_models', exist_ok=True)
    file_path = f'..\..\models\saved_models\baseline_{train_domain}.pkl'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    model_data = {
        'vectorizer': vectorizer,
        'classifier': classifier,
        'config': config
    }
        
    with open(file_path, 'wb') as f:
        pickle.dump(model_data, f)