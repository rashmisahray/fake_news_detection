import torch
import os
import pickle
from src.bilstm_attention import BiLSTMWithAttention
from src.bert_hybrid import HybridBERTModel
from src.feature_extractor import ClassicalFeatureExtractor
import torch
import os
import pickle
from src.bilstm_attention import BiLSTMWithAttention
from src.bert_hybrid import HybridBERTModel
from src.feature_extractor import ClassicalFeatureExtractor

os.makedirs('output', exist_ok=True)

# Define vocab size
VOCAB_SIZE = 5000

# Create a dummy TF-IDF vectorizer
print("Generating dummy TF-IDF vectorizer with 100 features...")
extractor = ClassicalFeatureExtractor(max_tfidf_features=100)
# Use a wide variety of words to ensure we hit the 100 feature limit
dummy_texts = [
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega",
    "red blue green yellow orange purple pink brown black white gray silver gold copper bronze iron steel lead tin zinc",
    "apple banana cherry date elderberry fig grape honeydew kiwi lemon mango nectarine orange papaya quince raspberry strawberry tangerine uva watermelon",
    "The quick brown fox jumps over the lazy dog",
    "Misinformation detection systems use BERT and LSTM models to analyze linguistic patterns and sensationalism markers in news content.",
    "Official reports from government agencies provide trusted data points for verification against knowledge graphs and entity databases."
]
extractor.fit_tfidf(dummy_texts)
extractor.save('output/tfidf_vectorizer.pkl')

# Calculate exact dim for the saved model
classical_dim = extractor.tfidf.get_feature_names_out().shape[0] + 7
print(f"Generated extractor dim: {classical_dim}")

print(f"Saving weights with vocab size {VOCAB_SIZE} and classical_dim {classical_dim}...")
bilstm = BiLSTMWithAttention(VOCAB_SIZE, 128, 256, 2)
torch.save(bilstm.state_dict(), 'output/bilstm_attn.pth')

bert = HybridBERTModel(classical_dim)
torch.save(bert.state_dict(), 'output/bert_hybrid.pth')

# Create a vocab that matches the size
word_to_idx = {f"word_{i}": i+2 for i in range(VOCAB_SIZE - 2)}
word_to_idx['<PAD>'] = 0
word_to_idx['<UNK>'] = 1

with open('output/word_to_idx.pkl', 'wb') as f:
    pickle.dump(word_to_idx, f)

print("Done. All model resources generated in /output.")
