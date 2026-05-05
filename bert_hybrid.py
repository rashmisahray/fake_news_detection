import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW
from datasets import load_dataset
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp

nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

def get_classical_features(texts):
    # 1. TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # 2. POS Adjective-to-noun ratio
    adj_noun_ratios = []
    # 3. Sentiment polarity
    polarities = []
    
    for text in tqdm(texts, desc="Extracting Classical Features"):
        blob = TextBlob(text)
        polarities.append(blob.sentiment.polarity)
        
        pos_tags = [pos for word, pos in blob.tags]
        num_adjectives = sum(1 for tag in pos_tags if tag.startswith('JJ'))
        num_nouns = sum(1 for tag in pos_tags if tag.startswith('NN'))
        
        ratio = num_adjectives / num_nouns if num_nouns > 0 else 0
        adj_noun_ratios.append(ratio)
        
    # Combine
    adj_noun_ratios = np.array(adj_noun_ratios).reshape(-1, 1)
    polarities = np.array(polarities).reshape(-1, 1)
    
    # Add to sparse matrix
    from scipy.sparse import hstack
    classical_features = hstack([tfidf_matrix, sp.csr_matrix(adj_noun_ratios), sp.csr_matrix(polarities)])
    
    return classical_features, vectorizer

def get_test_classical_features(texts, vectorizer):
    tfidf_matrix = vectorizer.transform(texts)
    adj_noun_ratios = []
    polarities = []
    
    for text in texts:
        blob = TextBlob(text)
        polarities.append(blob.sentiment.polarity)
        pos_tags = [pos for word, pos in blob.tags]
        num_adjectives = sum(1 for tag in pos_tags if tag.startswith('JJ'))
        num_nouns = sum(1 for tag in pos_tags if tag.startswith('NN'))
        ratio = num_adjectives / num_nouns if num_nouns > 0 else 0
        adj_noun_ratios.append(ratio)
        
    adj_noun_ratios = np.array(adj_noun_ratios).reshape(-1, 1)
    polarities = np.array(polarities).reshape(-1, 1)
    from scipy.sparse import hstack
    classical_features = hstack([tfidf_matrix, sp.csr_matrix(adj_noun_ratios), sp.csr_matrix(polarities)])
    return classical_features

class HybridDataset(Dataset):
    def __init__(self, texts, classical_feats, labels, tokenizer, max_len=128):
        self.texts = texts
        self.classical_feats = classical_feats.toarray() if sp.issparse(classical_feats) else classical_feats
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'classical_features': torch.tensor(self.classical_feats[idx], dtype=torch.float),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

class HybridModel(nn.Module):
    def __init__(self, classical_dim, use_bert=True, use_classical=True):
        super(HybridModel, self).__init__()
        self.use_bert = use_bert
        self.use_classical = use_classical
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Freeze BERT for faster training if just hybrid testing
        for param in self.bert.parameters():
            param.requires_grad = False
            
        combined_dim = 0
        if use_bert:
            combined_dim += self.bert.config.hidden_size # 768
        if use_classical:
            combined_dim += classical_dim
            
        self.fc1 = nn.Linear(combined_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, classical_features):
        features = []
        if self.use_bert:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            features.append(pooled_output)
            
        if self.use_classical:
            features.append(classical_features)
            
        if len(features) > 1:
            combined = torch.cat(features, dim=1)
        else:
            combined = features[0]
            
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)

def train_hybrid_model(model, train_loader, val_loader, epochs=3, device='cpu'):
    criterion = nn.BCELoss()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            classical_features = batch['classical_features'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, classical_features)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
def evaluate(model, loader, device):
    model.eval()
    preds = []
    actuals = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            classical_features = batch['classical_features'].to(device)
            labels = batch['labels'].numpy()
            
            outputs = model(input_ids, attention_mask, classical_features).squeeze().cpu().numpy()
            preds.extend(outputs)
            actuals.extend(labels)
            
    binary_preds = [1 if p > 0.5 else 0 for p in preds]
    return f1_score(actuals, binary_preds), accuracy_score(actuals, binary_preds), binary_preds, actuals

def main():
    print("Loading dataset...")
    dataset = load_dataset('liar', trust_remote_code=True)
    
    # Use smaller subset for speed in dev
    # For full assignment, you would use full sets. Here we use 2000 samples to make training fast
    train_texts = dataset['train']['statement'][:2000]
    train_labels = [1 if l in [3, 4, 5] else 0 for l in dataset['train']['label'][:2000]]
    
    test_texts = dataset['test']['statement'][:500]
    test_labels = [1 if l in [3, 4, 5] else 0 for l in dataset['test']['label'][:500]]
    
    print("Extracting classical features for train...")
    train_classical_features, vectorizer = get_classical_features(train_texts)
    
    print("Extracting classical features for test...")
    test_classical_features = get_test_classical_features(test_texts, vectorizer)
    
    # Model (b) Classical Alone (using Scikit-Learn for simplicity, or neural network)
    print("\n--- Training Classical ML Model ---")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_classical_features, train_labels)
    classical_preds = clf.predict(test_classical_features)
    f1_classical = f1_score(test_labels, classical_preds)
    print(f"Classical Features Alone F1-Score: {f1_classical:.4f}")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_dataset = HybridDataset(train_texts, train_classical_features, train_labels, tokenizer)
    test_dataset = HybridDataset(test_texts, test_classical_features, test_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classical_dim = train_classical_features.shape[1]
    
    print("\n--- Training BERT Alone ---")
    model_bert = HybridModel(classical_dim, use_bert=True, use_classical=False)
    train_hybrid_model(model_bert, train_loader, test_loader, epochs=2, device=device)
    f1_bert, _, _, _ = evaluate(model_bert, test_loader, device)
    print(f"BERT Alone F1-Score: {f1_bert:.4f}")
    
    print("\n--- Training Hybrid Model ---")
    model_hybrid = HybridModel(classical_dim, use_bert=True, use_classical=True)
    train_hybrid_model(model_hybrid, train_loader, test_loader, epochs=2, device=device)
    f1_hybrid, acc_hybrid, hybrid_preds, actuals = evaluate(model_hybrid, test_loader, device)
    print(f"Hybrid Model F1-Score: {f1_hybrid:.4f}")
    
    print("\n--- Error Analysis ---")
    # Identify 20 misclassified examples
    misclassified = []
    for i, (pred, actual) in enumerate(zip(hybrid_preds, actuals)):
        if pred != actual:
            misclassified.append({
                'text': test_texts[i],
                'predicted': "Real" if pred == 1 else "Fake",
                'actual': "Real" if actual == 1 else "Fake"
            })
        if len(misclassified) == 20:
            break
            
    print(f"Found {len(misclassified)} misclassified examples for analysis:")
    for i, err in enumerate(misclassified):
        print(f"\nExample {i+1}:")
        print(f"Text: {err['text']}")
        print(f"Predicted: {err['predicted']}, Actual: {err['actual']}")

if __name__ == "__main__":
    main()
