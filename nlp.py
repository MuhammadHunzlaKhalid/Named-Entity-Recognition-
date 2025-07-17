import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import os

# --------------------
# 1. Load the dataset
# --------------------

# Try to load CSV with fallback to Excel
file_path = "NER dataset.csv"  # change to .xlsx if needed
if file_path.endswith(".csv"):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 decode error! Trying cp1252 encoding...")
        df = pd.read_csv(file_path, encoding='cp1252')
else:
    # If it's Excel or other formats
    df = pd.read_excel(file_path)

print("✅ Data loaded successfully!")

# --------------------
# 2. Preprocess the data
# --------------------

df['Sentence #'] = df['Sentence #'].ffill()

# Group words and tags by sentence
sentences = df.groupby("Sentence #")["Word"].apply(list).tolist()
tags = df.groupby("Sentence #")["Tag"].apply(list).tolist()

# --------------------
# 3. Create vocab and tag mappings
# --------------------

words = list(set(df["Word"].tolist()))
tags_flat = list(set(df["Tag"].tolist()))

word2idx = {w: i + 2 for i, w in enumerate(words)}  # start from 2
word2idx["PAD"] = 0
word2idx["UNK"] = 1

tag2idx = {t: i for i, t in enumerate(tags_flat)}
idx2tag = {i: t for t, i in tag2idx.items()}

# Debug print
print(f"Vocab size (word2idx): {len(word2idx)}")
print(f"Number of tags: {len(tag2idx)}")

# --------------------
# 4. Encode sentences and tags
# --------------------

X = [[word2idx.get(w, word2idx["UNK"]) for w in s] for s in sentences]
y = [[tag2idx[t] for t in ts] for ts in tags]

# --------------------
# 5. Pad sequences
# --------------------

def pad(sequences, pad_value=0):
    max_len = max(len(seq) for seq in sequences)
    return [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]

X_padded = pad(X, word2idx["PAD"])
y_padded = pad(y, tag2idx["O"])  # assuming 'O' is used for non-entity tokens

# Debug prints for index ranges
max_index = max([max(seq) for seq in X_padded])
print(f"Max index in data: {max_index}")

# --------------------
# 6. Convert to PyTorch tensors
# --------------------

X_tensor = torch.tensor(X_padded, dtype=torch.long)
y_tensor = torch.tensor(y_padded, dtype=torch.long)

# --------------------
# 7. Split into train/test
# --------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42
)

# --------------------
# 8. Create Dataset and DataLoader
# --------------------

class NERDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(NERDataset(X_train, y_train), batch_size=16, shuffle=True)

# --------------------
# 9. Define the BiLSTM model
# --------------------

class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=128):
        super(BiLSTM_NER, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=word2idx["PAD"])
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        out = self.fc(lstm_out)
        return out

# --------------------
# 10. Initialize model
# --------------------

vocab_size = max(word2idx.values()) + 1  # fix index out-of-range error
model = BiLSTM_NER(vocab_size=vocab_size, tagset_size=len(tag2idx))
criterion = nn.CrossEntropyLoss(ignore_index=tag2idx["O"])  # ignore 'O' label during loss calculation
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --------------------
# 11. Training loop
# --------------------

num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"🚀 Using device: {device}")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_X)
        output = output.view(-1, output.shape[-1])
        batch_y = batch_y.view(-1)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

# --------------------
# 12. Save the model
# --------------------

os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "ner_bilstm_model.pth")
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved to: {model_path}")

print("🎉 Training complete!")
