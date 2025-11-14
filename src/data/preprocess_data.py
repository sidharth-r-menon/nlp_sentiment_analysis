from sklearn.model_selection import train_test_split
import pandas as pd
import os
import pickle
import re
import html
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

datasets_domain = {'imdb', 'amazon', 'yelp'}
raw_data_directory = "data/raw"
processed_data_directory = "data/processed"
test_size_percentage = 0.2
valuation_size_percentage = 0.1
random_seed = 42

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

all_data = {}

for domain in datasets_domain:
    df = pd.DataFrame()
    if domain == 'imdb':
        df = pd.read_csv('imdb/IMDB Dataset.csv', encoding='utf-8')
        df = df.rename(columns={'review': 'text', 'sentiment': 'label'})
        df['label'] = df['label'].map({'negative': 0, 'positive': 1})

    elif domain == 'amazon':
        data = []
        
        for file_path in ['amazon/train.ft.txt', 'amazon/test.ft.txt']:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    label = 1 if "__label__2" in line else 0 #__label__1 (negative)-> 0, __label__2 (positive) -> 1
                    text = (
                        line.replace("__label__1", "")
                        .replace("__label__2", "")
                        .strip()
                    )        
                    data.append({'text': text, 'label': label})
        
        df = pd.DataFrame(data)
    elif domain == 'yelp':
        dfs = []
        
        for file_path in ['yelp/train.csv', 'yelp/test.csv']:
            df = pd.read_csv(file_path, header=None, encoding='utf-8', names=["label", "text"])
            df['label'] = df['label'].map({1: 0, 2: 1}) # 1->0 (negative), 2->1 (positive)
            dfs.append(df)
        
        df = pd.concat(dfs, ignore_index=True)

    #remove missing values
    df = df.dropna()

    train_valuation, test = train_test_split(
        df,
        test_size=test_size_percentage,
        random_state=random_seed,
        stratify = df['label']
    )

    val_size = valuation_size_percentage / (1 - test_size_percentage)
    
    train, val = train_test_split(
        train_valuation,
        test_size=val_size,
        random_state=random_seed,
        stratify=train_valuation['label']
    )    
    
    all_data[domain] = {
        'train': train,
        'val': val,
        'test': test,
    }

def preprocess_text(text, remove_html=True, remove_urls=True, lowercase=True,
                    remove_punctuation=False, remove_stopwords=False, lemmatize=False):        
    if remove_html:
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', ' ', text)
    if remove_urls:
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    if lowercase:
        text = text.lower()
    
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', ' ', text)     
    
    if remove_stopwords:
        tokens = text.split()
        filtered_tokens = [w for w in tokens if w.lower() not in stop_words]
        text = ' '.join(filtered_tokens)
    
    if lemmatize:
        tokens = text.split()
        lemmatized = [lemmatizer.lemmatize(w) for w in tokens]
        text = ' '.join(lemmatized)
                
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text
    
preprocessed_data = {}

for domain, splits in all_data.items():
    preprocessed_data[domain] = {}
    
    for split_name, data_df in splits.items():
        data_copied_df = data_df.copy()
        data_copied_df['original_text'] = data_copied_df['text']
        data_copied_df['text'] = data_copied_df['text'].apply(preprocess_text)
        data_copied_df = data_copied_df[data_copied_df['text'].str.len() > 0]
        preprocessed_data[domain][split_name] = data_copied_df
        
os.makedirs(processed_data_directory, exist_ok=True)

output_path = os.path.join(processed_data_directory, 'preprocessed_data.pkl')
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'wb') as f:
    pickle.dump(preprocessed_data, f)

print(f"Preprocessed data saved to {output_path}")