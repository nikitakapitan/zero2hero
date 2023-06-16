"""Neuro-based Bigram LM

The difference with vanilla BLM:
• torch.nn.Embeddings (token emb)
+ torch.nn.Embeddings (positional emb)
+ nn.Linear
"""

import numpy as np
import torch

BATCH_SIZE = 32
BLOCK_SIZE = 8      # conetxt lenght
MAX_ITERS = 3000    # aka train epochs
EVAL_ITERS = 200    # how many batches to average for eval loss estimate
EVAL_INTERVAL = 300 # how often to compute eval loss estimate
LR = 1e-2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMB_SIZE = 32

# open shakespear
with open('docs/shakespear.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

encode = lambda s : [stoi[c] for c in s]
decode = lambda idx : ''.join([itos[i] for i in idx])

data = torch.tensor(encode(text), dtype=torch.long)

split_ratio = int(0.9 * len(data))
train_data = data[:split_ratio]
val_data = data[split_ratio:]

def get_batch(data):
    idx = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,)) # (BATCH)
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in idx]) # (BATCH, BLOCK)
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in idx]) # (BATCH, BLOCK)
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x,y

@torch.no_grad()
def estimate_loss():
    model.eval()
    
    train_losses = torch.zeros(EVAL_ITERS)
    for k in range(EVAL_ITERS):
        X, Y = get_batch(train_data)
        _, loss = model(X, Y)
        train_losses[k] = loss
    # val
    val_losses = torch.zeros(EVAL_ITERS)
    for k in range(EVAL_ITERS):
        X, Y = get_batch(val_data)
        _, loss = model(X, Y)
        val_losses[k] = loss
    model.train()
    return {'train': train_losses.mean(), 'val': val_losses.mean()}
        

class BigramLanguageModel(torch.nn.Module):
    # Neuro-based BLM

    def __init__(self, vocab_size, emb_size):
        super().__init__()
        # embed token value to some low-lev representation
        self.token_emb_table = torch.nn.Embedding(vocab_size, emb_size)
        # embed token position to some low-lev representation
        self.position_emb_table = torch.nn.Embedding(BLOCK_SIZE, emb_size)

        self.fc = torch.nn.Linear(emb_size, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape # (B, T)

        tok_emb = self.token_emb_table(idx) # (Batch, Time, Emb)
        # pos_emb = self.position_emb_table(torch.arange(T, device=DEVICE)) # (Time,Emb)
        # broadcast pos_emb to (•, Time, Emb)
        # pos_tok_emb = tok_emb + pos_emb # (Batch, Time, Emb)
        logits = self.fc(tok_emb) # (Batch, Time, Vocab)

        if targets is not None:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """
        idx (B, T)
        """
        for _ in range(max_new_tokens):

            logits, loss = self(idx) # (B, T, C)

            # context : use only last char
            logits = logits[:, -1, :] # (B, C)
            
            probs = torch.nn.functional.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1) 
        return idx
    
model = BigramLanguageModel(vocab_size=vocab_size, emb_size=EMB_SIZE)
model = model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# train
for iter in range(MAX_ITERS):

    if iter % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f'{iter=} {losses}')

    xb, yb = get_batch(train_data)

    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# generate
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
result = model.generate(context, max_new_tokens=100)[0].tolist()
print(decode(result))

    