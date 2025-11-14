import pandas as pd
import os, pickle, torch, sys 
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from inference_preprocessor import InferencePreprocessor


def loaddata(p):
    with open(p, 'rb') as f:
        data_raw = pickle.load(f)
    t_data = {}
    test_domains = ["imdb", "amazon", "yelp"]
    for d in test_domains:
        d_test = data_raw[d]['test']
        if isinstance(d_test, pd.DataFrame):
            x, y = d_test['text'].tolist(), d_test['label'].tolist()
        else:
            x, y = d_test[0], d_test[1]
        t_data[d] = {"x_processed": x, "y": y}
    return t_data

def metrics(y_t, preds, probs, m_type, train_d, test_d):
    acc = accuracy_score(y_t, preds)
    rep = classification_report(y_t, preds, target_names=["0","1"], output_dict=True)
    auc_in = probs if len(probs) else preds
    try:
        roc_auc = roc_auc_score(y_t, auc_in)
    except:
        roc_auc = 0.0
    try:
        tn, fp, fn, tp = confusion_matrix(y_t, preds).ravel()
    except:
        tn = fp = fn = tp = 0

    out = {
        "model": m_type.capitalize(),
        "train_domain": train_d,
        "test_domain": test_d,
        "is_in_domain": (train_d == test_d),
        "accuracy": acc,
        "precision": rep['1']['precision'],
        "recall": rep['1']['recall'],
        "roc_auc": roc_auc,
        "f1": rep['1']['f1-score'],
        "tn": int(tn), 
        "fp": int(fp),
        "fn": int(fn), 
        "tp": int(tp),
        "specificity": rep['0']['recall'],
        "sensitivity": rep['1']['recall']
    }
    return out
data = loaddata("../../data/processed/preprocessed_data.pkl")
m_config = {
    "baseline-imdb": {"type": "baseline", "file": "baseline_imdb.pkl"},
    "baseline-amazon": {"type": "baseline", "file": "baseline_amazon.pkl"},
    "baseline-yelp": {"type": "baseline", "file": "baseline_yelp.pkl"},
    "bert-imdb": {"type": "transformer", "file": "bert_imdb.pt", "hf_name": "bert-base-uncased"},
    "bert-amazon": {"type": "transformer", "file": "bert_amazon.pt", "hf_name": "bert-base-uncased"},
    "bert-yelp": {"type": "transformer", "file": "bert_yelp.pt", "hf_name": "bert-base-uncased"},
    "distilbert-imdb": {"type": "transformer", "file": "distilbert_imdb.pt", "hf_name": "distilbert-base-uncased"},
    "distilbert-amazon": {"type": "transformer", "file": "distilbert_amazon.pt", "hf_name": "distilbert-base-uncased"},
    "distilbert-yelp": {"type": "transformer", "file": "distilbert_yelp.pt", "hf_name": "distilbert-base-uncased"},
    "roberta-imdb": {"type": "transformer", "file": "roberta_imdb.pt", "hf_name": "roberta-base"},
    "roberta-amazon": {"type": "transformer", "file": "roberta_amazon.pt", "hf_name": "roberta-base"},
    "roberta-yelp": {"type": "transformer", "file": "roberta_yelp.pt", "hf_name": "roberta-base"},
}


r_lists = {"baseline": [], "bert": [], "distilbert": [], "roberta": []} 
for MODEL_KEY, model_info in m_config.items():
    m_type = model_info["type"]
    m_file = model_info["file"]
    train_d = MODEL_KEY.split('-')[1] 
    model_name = MODEL_KEY.split('-')[0]

    prep = InferencePreprocessor() 
    
    mdir = os.path.join("..", "..", "models", "saved_models")
    model_path = os.path.join(mdir, m_file)

    model_assets = None
    if m_type == "baseline":
        with open(model_path, 'rb') as f:
            model_assets = pickle.load(f)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_assets_dict = torch.load(model_path, map_location=device, weights_only=True)
        prep.set_transformer_tokenizer(
            model_name=model_info['hf_name'],
            model_state_dict=model_assets_dict['model_state_dict']
        )

    for test_d in data.keys():
        x_processed = data[test_d]["x_processed"]
        y_t = data[test_d]["y"]
        preds, probs = [], []

        x_clean = x_processed

        if m_type == "baseline":
            x_vec = model_assets['vectorizer'].transform(x_clean) 
            preds = model_assets['classifier'].predict(x_vec)
            probs = model_assets['classifier'].predict_proba(x_vec)[:, 1]
        else:
            bs = 128 
            for i in range(0, len(x_clean), bs):
                batch_clean_text = x_clean[i : i + bs]
                encodings = prep.encode_texts(batch_clean_text)
                preds_batch, prob_batch, _ = prep.token_predict(
                    encodings['input_ids'], 
                    encodings['attention_mask']
                )
                preds.extend(preds_batch.numpy())
                probs.extend(prob_batch.numpy()[:, 1])
        
        result_row = metrics(y_t, preds, probs, m_type, train_d, test_d)
        r_lists[model_name].append(result_row)

    if m_type == "transformer":
        del prep.model
        torch.cuda.empty_cache()

metrics_dir = os.path.join("..", "..", "metrics")
os.makedirs(metrics_dir, exist_ok=True)
for model_name, results in r_lists.items():
    df = pd.DataFrame(results)
    out_csv = os.path.join(metrics_dir, f"{model_name}_results.csv")
    df.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"Saved {out_csv}")