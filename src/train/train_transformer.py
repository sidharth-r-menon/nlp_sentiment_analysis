import pickle
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

config = {
    "models": {
        "transformers": {
            "bert": {"model_name": "bert-base-uncased", "num_labels": 2},
            "roberta": {"model_name": "roberta-base", "num_labels": 2},
            "distilbert":{"model_name": "distilbert-base-uncased", "num_labels": 2},
        }
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
        "dataloader_num_workers": 0,
        "pin_memory": True,
    },

    "evaluation": {
        "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"]
    },
}


data = None

with open(r'..\..\data\processed\preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

domains = list(data.keys())
models = ['bert', 'roberta', 'distilbert']
seed = 42
batch_size = config['training']['batch_size']
num_workers = config['training']['dataloader_num_workers']
pin_memory = config['training']['pin_memory']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

np.random.seed(seed)
torch.manual_seed(seed)

def make_dataset(texts, labels, tokenizer):
    encodings = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    dataset = TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        torch.tensor(labels, dtype=torch.long)
    )
    return dataset

def train_epoch(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    progress_bar = tqdm(train_loader, desc="Training")

    for batch in progress_bar:
        # Handle both dict-style and tuple-style batches
        if isinstance(batch, dict):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
        else:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['training']['max_grad_norm'])
        optimizer.step()

        scheduler.step()

        predictions = torch.argmax(logits, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)
        total_loss += loss.item()

        progress_bar.set_postfix({
            'loss': loss.item(),
            'acc': correct_predictions / total_predictions
        })

    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions

    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

for model in models:
    for domain in domains:
        model_full_name = config['models']['transformers'][model]['model_name']
        tokenizer = AutoTokenizer.from_pretrained(model_full_name)

        domain_data = data[domain]
        train_texts = domain_data['train']['text'].tolist()
        train_labels = domain_data['train']['label'].tolist()

        valuation_texts = domain_data['val']['text'].tolist()
        valuation_labels = domain_data['val']['label'].tolist()

        train_dataset = make_dataset(
            train_texts, train_labels, tokenizer
        )
        val_dataset = make_dataset(
            valuation_texts, valuation_labels, tokenizer
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False
        )
        
        valuation_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False
        )

        transformer_model = AutoModelForSequenceClassification.from_pretrained(
            model_full_name,
            num_labels=2,
            output_attentions=True,
            output_hidden_states=False
        )

        transformer_model.to(device)

        num_training_steps = len(train_dataloader) * config['training']['num_epochs']
        optimizer = AdamW(
            transformer_model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['training']['warmup_steps'],
            num_training_steps=num_training_steps
        )

        for epoch in range(config['training']['num_epochs']):
            print(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
            train_metrics = train_epoch(transformer_model, train_dataloader, optimizer, scheduler, device)

        os.makedirs(r"..\..\models\saved_models", exist_ok=True)
        model_path = os.path.join(r"..\..\models\saved_models", f'{model}_{domain}.pt')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({
            'model_state_dict': transformer_model.state_dict(),
            'model_name': model,
            'config': config
        }, model_path)
