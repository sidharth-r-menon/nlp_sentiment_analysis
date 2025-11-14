import pandas as pd
import os, pickle, torch
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from evaluation.inference_preprocessor import InferencePreprocessor

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

def metrics_base(y_t, preds, probs, m_name, train_d, test_d):
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
        "model": m_name,
        "train_domain": train_d,
        "test_domain": test_d,
        "is_in_domain": (train_d == test_d),
        "accuracy": acc,
        "precision": rep['1']['precision'],
        "recall": rep['1']['recall'],
        "f1": rep['1']['f1-score'],
        "roc_auc": roc_auc,
        "tn": int(tn), 
        "fp": int(fp),
        "fn": int(fn), 
        "tp": int(tp),
        "specificity": rep['0']['recall'],
        "sensitivity": rep['1']['recall']
    }
    return out

def metrics_ensemble_summary(test_d, ensemble_metrics):
    out = {
        "test_domain": test_d,
        "ensemble_accuracy": ensemble_metrics['acc'],
        "ensemble_precision": ensemble_metrics['prec'],
        "ensemble_recall": ensemble_metrics['rec'],
        "ensemble_f1": ensemble_metrics['f1'],
        "ensemble_roc_auc": ensemble_metrics['roc_auc'],
        "best_base_f1": None, 
        "improvement": None,
        "imdb_f1": None, 
        "amazon_f1": None, 
        "yelp_f1": None 
    }
    return out

data = loaddata("data/preprocessed_data.pkl") 

m_configs = {
    "ensemble": {
        "bert": {"file": "bert_ensemble.pkl"},
        "distilbert": {"file": "distilbert_ense.pkl"},
        "roberta": {"file": "roberta_ensem.pkl"},
    }
}

# prep_global = InferencePreprocessor()

# for family_name, m_config in m_configs.items():
    
#     if family_name == 'ensemble':
#         continue

#     r_list_family = []
    
#     for MODEL_KEY, model_info in m_config.items():
#         m_type = model_info["type"]
#         m_file = model_info["file"]
#         train_d = MODEL_KEY.split('-')[1] 

#         model_family = MODEL_KEY.split('-')[0]
#         m_display_name = model_family.upper()
#         if model_family == 'baseline':
#             m_display_name = 'TF-IDF + LogReg'

#         prep = prep_global
#         if m_type == "transformer":
#             prep = InferencePreprocessor() 
        
#         mdir = "saved_models"
#         model_path = os.path.join(mdir, m_file)

#         model_assets = None
#         if m_type == "baseline":
#             with open(model_path, 'rb') as f:
#                 model_assets = pickle.load(f)
#         else:
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             model_assets_dict = torch.load(model_path, map_location=device, weights_only=True)
#             prep.set_transformer_tokenizer(
#                 model_name=model_info['hf_name'],
#                 model_state_dict=model_assets_dict['model_state_dict']
#             )

#         for test_d in data.keys():
#             x_processed = data[test_d]["x_processed"]
#             y_t = data[test_d]["y"]
#             preds, probs = [], []

#             x_clean = x_processed
            
#             if m_type == "baseline":
#                 x_vec = model_assets['vectorizer'].transform(x_clean) 
#                 preds = model_assets['classifier'].predict(x_vec)
#                 probs = model_assets['classifier'].predict_proba(x_vec)[:, 1]
#             else:
#                 bs = 128 
#                 for i in range(0, len(x_clean), bs):
#                     batch_clean_text = x_clean[i : i + bs]
#                     encodings = prep.encode_texts(batch_clean_text)
#                     preds_batch, prob_batch, _ = prep.token_predict(
#                         encodings['input_ids'], 
#                         encodings['attention_mask']
#                     )
#                     preds.extend(preds_batch.numpy())
#                     probs.extend(prob_batch.numpy()[:, 1])
            
#             result_row = metrics_base(y_t, preds, probs, m_display_name, train_d, test_d)
#             r_list_family.append(result_row)

#         if m_type == "transformer":
#             del prep.model
#             torch.cuda.empty_cache()

#     out_csv = f"{family_name}_results.csv"
#     df = pd.DataFrame(r_list_family)
#     df.to_csv(out_csv, index=False, encoding='utf-8')

prep_ensemble = InferencePreprocessor()

for ensemble_key, ensemble_info in m_configs['ensemble'].items():
    
    r_list_ensemble = []
    m_file = ensemble_info["file"]
    
    mdir = "saved_models"
    model_path = os.path.join(mdir, m_file)
    
    with open(model_path, 'rb') as f:
        model_assets = pickle.load(f)
        prep_ensemble.set_ensemble_data(model_assets)
    for test_d in data.keys():
        x_processed = data[test_d]["x_processed"]
        y_t = data[test_d]["y"]
        
        x_clean = x_processed

        domains = ['imdb', 'amazon', 'yelp']
        for domain in domains:
            model_path = f"saved_models/{ensemble_key}_{domain}.pt".lower() #fix
            domain_model_data = torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            prep_ensemble.add_base_model(domain, model_name=domain_model_data['model_name'], model_state_dict=domain_model_data['model_state_dict'])
        prep_ensemble.set_tokenizer(model_name=domain_model_data['model_name'])

        encodings = prep_ensemble.encode_texts(x_clean)
        with torch.no_grad():
            input_ids = encodings['input_ids']
            attention_mask = encodings['attention_mask']

            dataset = TensorDataset(
                input_ids,
                attention_mask,
                torch.tensor(y_t, dtype=torch.long)
            )
            loader = DataLoader(dataset, batch_size=64)
            results = prep_ensemble.ensemble_predict(loader)
            ensemble_prediction = results['predictions']
            ensemble_probability = results['probabilities']
        
        acc = accuracy_score(y_t, ensemble_prediction)
        rep = classification_report(y_t, ensemble_prediction, target_names=["0","1"], output_dict=True)
        try:
            roc_auc = roc_auc_score(y_t, ensemble_probability)
        except:
            roc_auc = 0.0

        ensemble_metrics = {
            'acc': acc, 'prec': rep['1']['precision'], 'rec': rep['1']['recall'],
            'f1': rep['1']['f1-score'], 'roc_auc': roc_auc
        }
        
        result_row = metrics_ensemble_summary(test_d, ensemble_metrics)
        r_list_ensemble.append(result_row)
        print(f"row completed {result_row}")
    out_csv = f"{ensemble_key}_ensemble_results.csv"
    df = pd.DataFrame(r_list_ensemble)
    df.to_csv(out_csv, index=False, encoding='utf-8')