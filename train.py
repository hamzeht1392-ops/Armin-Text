import os
import re
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import pre_tokenizers, decoders
from tqdm.auto import tqdm


# ---------------------------------
# Configuration
# ---------------------------------

SEED = 42

MAX_STORIES = 8000
MAX_WORDS = 600_000

VOCAB_SIZE = 12_000
SEQ_LEN = 48

BATCH_SIZE = 64
EPOCHS = 5

EMBEDDING_DIM = 192
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.25

LEARNING_RATE = 0.0015
WEIGHT_DECAY = 0.01

TOKENIZER_PATH = "tokenizer.json"
MODEL_PATH = "ArminText-Final.pt"


# ---------------------------------
# Reproducibility
# ---------------------------------

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU is not available. Training on CPU.")


# ---------------------------------
# Persian normalization
# ---------------------------------

def normalize_persian(text):
    text = text.replace("ي", "ی")
    text = text.replace("ى", "ی")
    text = text.replace("ك", "ک")
    text = text.replace("ة", "ه")
    text = text.replace("ۀ", "ه")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def contains_persian(text):
    if not isinstance(text, str):
        return False

    return any(
        "\u0600" <= ch <= "\u06FF"
        for ch in text
    )


# ---------------------------------
# Load dataset
# ---------------------------------

dataset = load_dataset(
    "taesiri/TinyStories-Farsi",
    split="train",
    streaming=True,
)

sample = next(iter(dataset))

persian_column = None

for key, value in sample.items():
    if contains_persian(value):
        persian_column = key
        break

if persian_column is None:
    raise ValueError("Persian text column was not found.")

print("Persian column:", persian_column)


# ---------------------------------
# Collect stories
# ---------------------------------

texts = []
total_words = 0

dataset = load_dataset(
    "taesiri/TinyStories-Farsi",
    split="train",
    streaming=True,
)

progress = tqdm(
    dataset,
    total=MAX_STORIES,
    desc="Collecting stories",
)

for row in progress:

    text = row[persian_column]

    if not isinstance(text, str):
        continue

    text = normalize_persian(text)

    if len(text) < 50:
        continue

    words = text.split()

    if len(words) < 10:
        continue

    if total_words + len(words) > MAX_WORDS:
        break

    texts.append(text)
    total_words += len(words)

    if len(texts) >= MAX_STORIES:
        break


random.shuffle(texts)

print("Stories:", len(texts))
print("Approx words:", total_words)


# ---------------------------------
# Train / validation split
# ---------------------------------

split_index = int(len(texts) * 0.9)

train_texts = texts[:split_index]
val_texts = texts[split_index:]

print("Train stories:", len(train_texts))
print("Validation stories:", len(val_texts))


# ---------------------------------
# Train ByteLevel BPE tokenizer
# ---------------------------------

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[BOS]",
    "[EOS]",
]

tokenizer = Tokenizer(
    BPE(unk_token="[UNK]")
)

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=False
)

tokenizer.decoder = decoders.ByteLevel()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=SPECIAL_TOKENS,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
)

tokenizer.train_from_iterator(
    train_texts,
    trainer=trainer,
    length=len(train_texts),
)

tokenizer.save(TOKENIZER_PATH)

print("Tokenizer saved:", TOKENIZER_PATH)

vocab_size = tokenizer.get_vocab_size()

PAD_ID = tokenizer.token_to_id("[PAD]")
UNK_ID = tokenizer.token_to_id("[UNK]")
BOS_ID = tokenizer.token_to_id("[BOS]")
EOS_ID = tokenizer.token_to_id("[EOS]")

print("Vocabulary size:", vocab_size)


# ---------------------------------
# Encoding
# ---------------------------------

def encode_text(text):
    token_ids = tokenizer.encode(text).ids

    return [
        BOS_ID,
        *token_ids,
        EOS_ID,
    ]


train_ids = []

for text in tqdm(
    train_texts,
    desc="Encoding train",
):
    train_ids.extend(encode_text(text))


val_ids = []

for text in tqdm(
    val_texts,
    desc="Encoding validation",
):
    val_ids.extend(encode_text(text))


print("Train tokens:", len(train_ids))
print("Validation tokens:", len(val_ids))


# ---------------------------------
# UNK statistics
# ---------------------------------

train_unk = train_ids.count(UNK_ID)
val_unk = val_ids.count(UNK_ID)

train_unk_percent = (
    train_unk / len(train_ids) * 100
)

val_unk_percent = (
    val_unk / len(val_ids) * 100
)

print(
    f"Train UNK percentage: "
    f"{train_unk_percent:.4f}%"
)

print(
    f"Validation UNK percentage: "
    f"{val_unk_percent:.4f}%"
)


# ---------------------------------
# Dataset
# ---------------------------------

class NextTokenDataset(Dataset):

    def __init__(self, token_ids, seq_len):
        self.data = torch.tensor(
            token_ids,
            dtype=torch.long,
        )

        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, index):

        x = self.data[
            index:index + self.seq_len
        ]

        y = self.data[
            index + 1:index + self.seq_len + 1
        ]

        return x, y


train_dataset = NextTokenDataset(
    train_ids,
    SEQ_LEN,
)

val_dataset = NextTokenDataset(
    val_ids,
    SEQ_LEN,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


print(
    "Train samples:",
    len(train_dataset),
)

print(
    "Validation samples:",
    len(val_dataset),
)


# ---------------------------------
# Model
# ---------------------------------

class ArminText(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            hidden_dim,
            vocab_size,
        )

    def forward(self, x):

        x = self.embedding(x)

        output, hidden = self.gru(x)

        output = self.dropout(output)

        logits = self.fc(output)

        return logits, hidden


model = ArminText(
    vocab_size=vocab_size
).to(device)


total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "Parameters:",
    f"{total_params:,}",
)


# ---------------------------------
# Training setup
# ---------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ---------------------------------
# Training loop
# ---------------------------------

best_val_loss = float("inf")
best_state = None

for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0.0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}",
    )

    for x, y in progress:

        x = x.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad()

        logits, _ = model(x)

        loss = criterion(
            logits.reshape(-1, vocab_size),
            y.reshape(-1),
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_train_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    avg_train_loss = (
        total_train_loss /
        len(train_loader)
    )

    # Validation
    model.eval()

    total_val_loss = 0.0

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(
                device,
                non_blocking=True,
            )

            y = y.to(
                device,
                non_blocking=True,
            )

            logits, _ = model(x)

            loss = criterion(
                logits.reshape(-1, vocab_size),
                y.reshape(-1),
            )

            total_val_loss += loss.item()

    avg_val_loss = (
        total_val_loss /
        len(val_loader)
    )

    print()

    print(
        f"Epoch {epoch + 1}: "
        f"Train Loss = {avg_train_loss:.4f} | "
        f"Val Loss = {avg_val_loss:.4f}"
    )

    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        best_state = {
            key: value.detach()
            .cpu()
            .clone()
            for key, value
            in model.state_dict().items()
        }

        print("New best model.")


# ---------------------------------
# Save best model
# ---------------------------------

if best_state is None:
    raise RuntimeError(
        "No model checkpoint was created."
    )


model.load_state_dict(best_state)
model = model.to(device)

torch.save(
    {
        "model_state_dict": model.state_dict(),

        "config": {
            "vocab_size": vocab_size,
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "seq_len": SEQ_LEN,
        },
    },
    MODEL_PATH,
)

print()
print("Model saved:", MODEL_PATH)
print(
    f"Best Validation Loss: "
    f"{best_val_loss:.4f}"
)
