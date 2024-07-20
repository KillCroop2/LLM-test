import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
import random
import math
from tqdm import tqdm
import os
import nltk
import multiprocessing
import argparse

# Ensure NLTK datasets are downloaded if needed
nltk.download('punkt', quiet=True)  # For tokenization

class EnhancedTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward=1024):
        super(EnhancedTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        return self.fc(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CustomTextDataset(Dataset):
    def __init__(self, filepath, word2idx, seq_length):
        self.filepath = filepath
        self.word2idx = word2idx
        self.seq_length = seq_length
        self.texts = self.load_data(filepath)

    def load_data(self, filepath):
        with open(filepath, 'r') as file:
            lines = file.readlines()
        # Tokenize and preprocess
        texts = []
        for line in lines:
            tokens = nltk.word_tokenize(line.strip().lower())
            texts.append(' '.join(tokens))
        return texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in text.split()]
        if len(encoded) < self.seq_length + 1:
            encoded += [self.word2idx['<PAD>']] * (self.seq_length + 1 - len(encoded))
        start_idx = random.randint(0, len(encoded) - self.seq_length - 1)
        src = torch.tensor(encoded[start_idx:start_idx + self.seq_length])
        tgt = torch.tensor(encoded[start_idx + 1:start_idx + self.seq_length + 1])
        return src, tgt

def generate_text(model, start_text, word2idx, idx2word, max_length=20, temperature=1.0):
    model.eval()
    words = start_text.split()
    device = next(model.parameters()).device
    current_ids = torch.tensor([word2idx.get(w, word2idx['<UNK>']) for w in words]).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(current_ids)
            next_word_logits = logits[0, -1, :] / temperature
            next_word = torch.multinomial(torch.softmax(next_word_logits, dim=-1), num_samples=1).item()
            current_ids = torch.cat([current_ids, torch.tensor([[next_word]]).to(device)], dim=1)
            words.append(idx2word.get(next_word, '<UNK>'))
            
            if words[-1] == '.':
                break
    
    return ' '.join(words)


def build_vocab(texts, vocab_size):
    word_counts = Counter(word for text in texts for word in text.lower().split())
    vocab = ['<PAD>', '<UNK>'] + [word for word, _ in word_counts.most_common(vocab_size - 2)]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, scheduler, filename):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded from {filename}")
        return epoch, loss
    return 0, float('inf')

def main(local_rank=None):
    # Hyperparameters
    vocab_size = 10000
    d_model = 512
    nhead = 16
    num_layers = 12
    batch_size = 64
    seq_length = 50
    num_epochs = 200
    learning_rate = 0.00005
    custom_dataset_path = './large_text_dataset.txt'
    checkpoint_filename = 'enhanced_transformer_checkpoint.pth'
    accumulation_steps = 4
    patience = 5

    # Detect number of GPUs and set up distributed training if applicable
    num_gpus = torch.cuda.device_count()
    is_distributed = num_gpus > 1

    if is_distributed:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Number of GPUs available: {num_gpus}")

    # Load custom dataset and build vocab only on rank 0
    if not is_distributed or local_rank == 0:
        dataset = CustomTextDataset(custom_dataset_path, word2idx=None, seq_length=seq_length)
        texts = dataset.texts
        word2idx, idx2word = build_vocab(texts, vocab_size)
    else:
        texts, word2idx, idx2word = None, None, None

    # Broadcast data to all processes
    if is_distributed:
        texts = [texts] if local_rank == 0 else [None]
        torch.distributed.broadcast_object_list(texts, src=0)
        texts = texts[0]

        word2idx = [word2idx] if local_rank == 0 else [None]
        torch.distributed.broadcast_object_list(word2idx, src=0)
        word2idx = word2idx[0]

        idx2word = [idx2word] if local_rank == 0 else [None]
        torch.distributed.broadcast_object_list(idx2word, src=0)
        idx2word = idx2word[0]

    # Create dataset and dataloader
    dataset = CustomTextDataset(custom_dataset_path, word2idx, seq_length)
    num_workers = min(multiprocessing.cpu_count(), 8)

    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True, prefetch_factor=1)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=1)

    # Create the model
    model = EnhancedTransformer(vocab_size, d_model, nhead, num_layers)
    model = model.to(device)
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # Load checkpoint if it exists
    start_epoch, best_loss = load_checkpoint(model, optimizer, scheduler, checkpoint_filename)

    # Initialize the GradScaler for mixed precision training
    scaler = GradScaler()

    # Training loop
    no_improvement = 0
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        if is_distributed:
            sampler.set_epoch(epoch)
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=(is_distributed and local_rank != 0))
        
        for batch, (src, tgt) in enumerate(progress_bar):
            src, tgt = src.to(device), tgt.to(device)
            
            with autocast():
                output = model(src)
                loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            
            if (batch + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            
            if not is_distributed or local_rank == 0:
                progress_bar.set_postfix({'loss': loss.item() * accumulation_steps})
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        if not is_distributed or local_rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(model.module if is_distributed else model, optimizer, scheduler, epoch + 1, best_loss, checkpoint_filename)
                no_improvement = 0
            else:
                no_improvement += 1
            
            if no_improvement >= patience:
                print(f"No improvement for {patience} epochs. Stopping training.")
                break

        # Clear CUDA cache to potentially free up memory
        torch.cuda.empty_cache()

    if not is_distributed or local_rank == 0:
        print("Training complete.")

        # Generate text
        print("\nGenerating sample texts:")
        start_texts = ["the quick brown", "a journey of", "to be or", "all that glitters", "where there is"]
        model_for_generation = model.module if is_distributed else model
        for start_text in start_texts:
            generated_text = generate_text(model_for_generation, start_text, word2idx, idx2word, max_length=50, temperature=0.8)
            print(f"\nStarting with '{start_text}':")
            print(generated_text)

        print("\nText generation complete.")