import os
import pickle
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def load_model(model_name, domain):
    model_path = f'../../models/saved_models/{model_name}_{domain}.pt'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model_configs = {
        'bert': 'bert-base-uncased',
        'roberta': 'roberta-base',
        'distilbert': 'distilbert-base-uncased'
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_path, map_location=device)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_configs[model_name], num_labels=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    
    return model, AutoTokenizer.from_pretrained(model_configs[model_name])


def get_predictions(model, tokenizer, texts, device, batch_size=32):
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
    
    return np.array(all_probs)


def train_ensemble(model_name, domains, data_path='../../data/processed/preprocessed_data.pkl'):
    print(f"\nTraining {model_name} ensemble with domains: {', '.join(domains)}")
    
    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load base models
    models, tokenizers = {}, {}
    for domain in domains:
        models[domain], tokenizers[domain] = load_model(model_name, domain)
    
    # Prepare meta-classifier training data using validation set from first domain
    train_domain = domains[0]
    val_texts = data[train_domain]['val']['text']
    val_labels = data[train_domain]['val']['label']
    
    if not isinstance(val_texts, list):
        val_texts = val_texts.tolist()
    if not isinstance(val_labels, list):
        val_labels = val_labels.tolist()
    
    meta_features = np.column_stack([
        get_predictions(models[d], tokenizers[d], val_texts, device)
        for d in domains
    ])
    
    # Train meta-classifier
    meta_clf = LogisticRegression(max_iter=1000, random_state=42)
    meta_clf.fit(meta_features, val_labels)
    
    # Evaluate on all domains
    results = []
    for test_domain in domains:
        test_texts = data[test_domain]['test']['text']
        test_labels = data[test_domain]['test']['label']
        
        if not isinstance(test_texts, list):
            test_texts = test_texts.tolist()
        if not isinstance(test_labels, list):
            test_labels = test_labels.tolist()
        
        test_features = np.column_stack([
            get_predictions(models[d], tokenizers[d], test_texts, device)
            for d in domains
        ])
        
        ensemble_preds = meta_clf.predict(test_features)
        ensemble_probs = meta_clf.predict_proba(test_features)[:, 1]
        
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels, ensemble_preds, average='binary')
        
        results.append({
            'domain': test_domain,
            'accuracy': accuracy_score(test_labels, ensemble_preds),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc_score(test_labels, ensemble_probs)
        })
        
        print(f"{test_domain}: Acc={results[-1]['accuracy']:.4f}, F1={results[-1]['f1']:.4f}")
    
    # Save results
    os.makedirs('../../metrics', exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'../../metrics/{model_name}_ensemble_results.csv', index=False)
    
    # Save ensemble
    os.makedirs('../../models/saved_models', exist_ok=True)
    ensemble_data = {'meta_clf': meta_clf, 'domains': domains}
    with open(f'../../models/saved_models/{model_name}_ensemble.pkl', 'wb') as f:
        pickle.dump(ensemble_data, f)
    
    print(f"\nAverage F1: {results_df['f1'].mean():.4f}")
    print(f"Results saved to: ../../metrics/{model_name}_ensemble_results.csv")
    
    return meta_clf, results_df


def main():
    models = ['bert', 'roberta', 'distilbert']
    domains = ['amazon', 'imdb', 'yelp']
    data_path = '../../data/processed/preprocessed_data.pkl'
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Training ensemble for {model.upper()}")
        train_ensemble(model, domains, data_path)


if __name__ == "__main__":
    main()
