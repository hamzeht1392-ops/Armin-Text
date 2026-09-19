import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ArminText(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=192,
        hidden_dim=256,
        num_layers=2,
        dropout=0.25,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, hidden = self.gru(x)
        output = self.dropout(output)
        logits = self.fc(output)

        return logits, hidden


# -----------------------------
# Load tokenizer and checkpoint
# -----------------------------

tokenizer = Tokenizer.from_file("tokenizer.json")

checkpoint = torch.load(
    "ArminText-Final.pt",
    map_location=device,
    weights_only=True,
)

config = checkpoint["config"]

vocab_size = config["vocab_size"]
seq_len = config["seq_len"]

PAD_ID = tokenizer.token_to_id("[PAD]")
BOS_ID = tokenizer.token_to_id("[BOS]")
EOS_ID = tokenizer.token_to_id("[EOS]")


# -----------------------------
# Build model
# -----------------------------

model = ArminText(
    vocab_size=vocab_size,
    embedding_dim=config["embedding_dim"],
    hidden_dim=config["hidden_dim"],
    num_layers=config["num_layers"],
    dropout=config["dropout"],
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# -----------------------------
# Text generation
# -----------------------------

def generate_text(
    prompt,
    max_new_tokens=60,
    temperature=0.8,
    top_k=30,
    repetition_penalty=1.1,
):
    prompt_ids = tokenizer.encode(prompt).ids

    generated = [BOS_ID] + prompt_ids

    with torch.no_grad():

        for _ in range(max_new_tokens):

            context = generated[-seq_len:]

            x = torch.tensor(
                [context],
                dtype=torch.long,
                device=device,
            )

            logits, _ = model(x)

            next_logits = logits[0, -1].clone()

            # Prevent special tokens
            next_logits[PAD_ID] = float("-inf")
            next_logits[BOS_ID] = float("-inf")

            # Reduce repetition
            recent_tokens = set(generated[-20:])

            for token_id in recent_tokens:
                if token_id not in [PAD_ID, BOS_ID]:
                    next_logits[token_id] /= repetition_penalty

            # Temperature
            next_logits /= temperature

            # Top-k sampling
            values, indices = torch.topk(
                next_logits,
                k=min(top_k, vocab_size),
            )

            probabilities = F.softmax(values, dim=-1)

            selected = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            next_token = indices[selected].item()

            if next_token == EOS_ID:
                break

            generated.append(next_token)

    output_ids = [
        token
        for token in generated
        if token not in [BOS_ID, EOS_ID, PAD_ID]
    ]

    return tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
    )


# -----------------------------
# Command line interface
# -----------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Generate Persian text with ArminText Final"
    )

    parser.add_argument(
        "prompt",
        type=str,
        help="Persian text prompt",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    result = generate_text(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print("\nPrompt:")
    print(args.prompt)

    print("\nArminText:")
    print(result)


if __name__ == "__main__":
    main()
