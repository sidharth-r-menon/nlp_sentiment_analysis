import re
import html
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
import torch
import shap

class InferencePreprocessor:

    def __init__(self):
        self.lowercase = True
        self.remove_html = True
        self.remove_urls = True
        self.remove_punctuation = False
        self.remove_stopwords = False
        self.lemmatize = False
        self.model_name = ""
        self.model = None
        self.tokenizer = None
        self.max_length = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.meta_classifier = None
        self.domains = ['imdb', 'amazon', 'yelp']
        self.is_fitted = False
        self.base_models = {}

    def _initialize_nltk(self):
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text):        
            if self.remove_html:
                text = html.unescape(text)
                text = re.sub(r'<[^>]+>', ' ', text)
            if self.remove_urls:
                text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

            if self.lowercase:
                text = text.lower()
            
            if self.remove_punctuation:
                text = re.sub(r'[^\w\s]', ' ', text)     
            
            if self.remove_stopwords:
                tokens = text.split()
                filtered_tokens = [w for w in tokens if w.lower() not in self.stop_words]
                text = ' '.join(filtered_tokens)
            
            if self.lemmatize:
                tokens = text.split()
                lemmatized = [self.lemmatizer.lemmatize(w) for w in tokens]
                text = ' '.join(lemmatized)
                        
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()

            return text
        
    def set_transformer_tokenizer(self, model_name, model_state_dict):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config = AutoConfig.from_pretrained(model_name, num_labels=2))
        self.model.load_state_dict(model_state_dict)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(self.device)
    
    def set_tokenizer(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def set_ensemble_data(self,ensemble_data):
        self.meta_classifier = ensemble_data['meta_classifier']
        self.domains = ensemble_data['domains']
        self.is_fitted = ensemble_data['is_fitted']
    
    def add_base_model(self, domain, model_name, model_state_dict):
        model= AutoModelForSequenceClassification.from_pretrained(model_name, config = AutoConfig.from_pretrained(model_name, num_labels=2))
        model.load_state_dict(model_state_dict)
        self.base_models[domain] = model

    def ensemble_predict(self, inputs):
        meta_input_features = []  # collect features from all base models

        for domain in self.domains:
            model = self.base_models[domain]
            model.to(self.device)
            model.eval()

            domain_probs = []

            with torch.no_grad():
                for batch in inputs:
                    # Each batch = (input_ids, attention_mask)
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits

                    probs = torch.softmax(logits, dim=1)[:, 1]
                    domain_probs.extend(probs.cpu().numpy())

            domain_probs = np.array(domain_probs)
            meta_input_features.append(domain_probs.reshape(-1, 1))
            meta_input_features.append((1 - domain_probs).reshape(-1, 1))

        meta_features = np.hstack(meta_input_features)

        y_pred = self.meta_classifier.predict(meta_features)
        y_prob = self.meta_classifier.predict_proba(meta_features)[:, 1]

        return {
            "predictions": y_pred,
            "probabilities": y_prob,
        }
    
    def ensemble_base_predictions(self, input_ids, attention_mask, domain):
        model = self.base_models[domain]
        model.eval()
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            attentions = outputs.attentions
            
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
        
        return predictions.cpu(), probabilities.cpu(), attentions

    def encode_texts(self, texts):
        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return encodings
    
    def token_predict(self, input_ids, attention_mask):
        self.model.eval()
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            attentions = outputs.attentions
            
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
        
        return predictions.cpu(), probabilities.cpu(), attentions

    def _predict_shap(self, unencoded_texts):
        if isinstance(unencoded_texts, np.ndarray):
            unencoded_texts = unencoded_texts.tolist()
            unencoded_texts = [str(t) for t in unencoded_texts]

        encodings = self.tokenizer(
            unencoded_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']

        _, probabilities, _ = self.token_predict(input_ids, attention_mask)
        
        return probabilities.numpy() 
    
    def get_shap_html(self, unencoded_texts):
        explainer = shap.Explainer(self._predict_shap, self.tokenizer)
        shap_values = explainer(unencoded_texts)
        return shap.plots.text(shap_values[0], display=False)
