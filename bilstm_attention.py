import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
import re
from bs4 import BeautifulSoup
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

class TextDataset(Dataset):
    def __init__(self, data, vocab, max_len=200):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['statement']
        # Binary label: fake/real. Liar has: 'half-true', 'false', 'mostly-true', 'barely-true', 'true', 'pants-fire'
        # Let's map mostly-true, half-true, true to 1 (Real)
        # false, barely-true, pants-fire to 0 (Fake)
        # Liar labels: 0: pants-fire, 1: false, 2: barely-true, 3: half-true, 4: mostly-true, 5: true
        label = 1 if item['label'] in [3, 4, 5] else 0

        # Preprocess
        text = text.lower()
        text = BeautifulSoup(text, "html.parser").get_text()
        tokens = word_tokenize(text)
        
        # Pad or truncate
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + ['<PAD>'] * (self.max_len - len(tokens))
            
        token_ids = [self.vocab.get(token, self.vocab.get('<UNK>')) for token in tokens]
        
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.float32), tokens

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        # Scaled dot-product self-attention
        self.W_q = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.W_k = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.W_v = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim * 2]))

    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_dim * 2)
        Q = self.W_q(lstm_output)
        K = self.W_k(lstm_output)
        V = self.W_v(lstm_output)
        
        # Attention scores: Q * K^T / sqrt(d)
        energy = torch.bmm(Q, K.transpose(1, 2)) / self.scale.to(Q.device)
        attention_weights = torch.softmax(energy, dim=-1)
        
        # Context vector: attention_weights * V
        context = torch.bmm(attention_weights, V)
        
        # Summarize sequence by summing/averaging context over seq_len
        # Let's just return the weights to visualize, and pool the context
        # For pooling, we can take the sum or max over the sequence dimension, or just use the last output
        # Here, let's use the attention over the whole sequence to get a single vector
        # A simpler attention for classification is often a global attention:
        # We can implement standard self-attention (Q=K=V=lstm_output)
        return context, attention_weights

class GlobalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(GlobalAttention, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1, bias=False)

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_dim*2)
        attn_weights = torch.softmax(self.attention(lstm_output).squeeze(2), dim=1) # (batch, seq_len)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1) # (batch, hidden_dim*2)
        return context, attn_weights

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, use_attention=True):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)
        self.use_attention = use_attention
        
        if use_attention:
            self.attention = GlobalAttention(hidden_dim)
            
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        if self.use_attention:
            context, attn_weights = self.attention(lstm_out)
            out = self.fc(context)
            return self.sigmoid(out), attn_weights
        else:
            # Use last hidden states for both directions
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            out = self.fc(hidden)
            return self.sigmoid(out), None

def build_vocab(data, max_vocab_size=20000):
    all_tokens = []
    for text in data['statement']:
        text = text.lower()
        text = BeautifulSoup(text, "html.parser").get_text()
        tokens = word_tokenize(text)
        all_tokens.extend(tokens)
    
    counter = Counter(all_tokens)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in counter.most_common(max_vocab_size):
        vocab[word] = len(vocab)
    return vocab

def plot_attention(tokens, attention_weights, filename):
    plt.figure(figsize=(15, 2))
    # Removing pads
    valid_len = 0
    for t in tokens:
        if t == '<PAD>':
            break
        valid_len += 1
        
    valid_len = min(valid_len, len(tokens))
    valid_tokens = tokens[:valid_len]
    valid_weights = attention_weights[:valid_len].cpu().detach().numpy()
    
    valid_weights = valid_weights.reshape(1, -1)
    
    sns.heatmap(valid_weights, xticklabels=valid_tokens, yticklabels=False, cmap='viridis', cbar=True)
    plt.xticks(rotation=45, ha='right')
    plt.title('Attention Weights for Fake News Classification')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def train_model(model, train_loader, val_loader, epochs=5, device='cpu'):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                val_preds.extend(outputs.squeeze().cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
        val_auc = roc_auc_score(val_labels, val_preds)
        val_acc = accuracy_score(val_labels, np.array(val_preds) > 0.5)
        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Val ACC: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

def main():
    print("Loading dataset...")
    dataset = load_dataset('liar', trust_remote_code=True)
    
    train_data = dataset['train']
    val_data = dataset['validation']
    test_data = dataset['test']
    
    # Take a subset to make it runnable quickly locally if needed, but let's use the full for now
    
    print("Building vocabulary...")
    vocab = build_vocab(train_data)
    
    print("Creating DataLoaders...")
    train_dataset = TextDataset(train_data, vocab)
    val_dataset = TextDataset(val_data, vocab)
    test_dataset = TextDataset(test_data, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train BiLSTM WITHOUT Attention
    print("\n--- Training BiLSTM (No Attention) ---")
    model_no_attn = BiLSTMClassifier(vocab_size=len(vocab), embed_dim=100, hidden_dim=256, use_attention=False)
    train_model(model_no_attn, train_loader, val_loader, epochs=3, device=device)
    
    # Train BiLSTM WITH Attention
    print("\n--- Training BiLSTM + Attention ---")
    model_attn = BiLSTMClassifier(vocab_size=len(vocab), embed_dim=100, hidden_dim=256, use_attention=True)
    train_model(model_attn, train_loader, val_loader, epochs=3, device=device)
    
    # Evaluate both on test set
    print("\n--- Test Set Evaluation ---")
    for name, m in [("BiLSTM", model_no_attn), ("BiLSTM+Attention", model_attn)]:
        m.eval()
        test_preds, test_labels = [], []
        test_tokens = []
        test_attns = []
        with torch.no_grad():
            for inputs, labels, tokens in test_loader:
                inputs = inputs.to(device)
                outputs, attns = m(inputs)
                test_preds.extend(outputs.squeeze().cpu().numpy())
                test_labels.extend(labels.numpy())
                if attns is not None:
                    test_attns.extend(attns)
                    test_tokens.extend(zip(*tokens)) # Unzip tokens for the batch
                    
        test_auc = roc_auc_score(test_labels, test_preds)
        test_acc = accuracy_score(test_labels, np.array(test_preds) > 0.5)
        print(f"{name} | Test ACC: {test_acc:.4f} | Test AUC: {test_auc:.4f}")
        
    # Generate Heatmap for one example
    if len(test_attns) > 0:
        # Find a true positive or just a random example
        idx = 0
        plot_attention(test_tokens[idx], test_attns[idx], "attention_heatmap.png")
        print("Saved attention heatmap to attention_heatmap.png")

if __name__ == "__main__":
    main()
