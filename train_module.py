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
import json
import io

# Ensure NLTK datasets are downloaded if needed
nltk.download('punkt', quiet=True)  # For tokenization

class EnhancedTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward=3072, dropout=0.2):
        super(EnhancedTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.dropout(output)
        return self.fc(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class JSONLDataset(Dataset):
    def __init__(self, filepath, word2idx, seq_length):
        self.filepath = filepath
        self.word2idx = word2idx
        self.seq_length = seq_length
        self.data = self.load_data(filepath)
        if self.word2idx is not None:
            self.encoded_texts = self.encode_texts()
        
    def load_data(self, filepath):
        data = []
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                json_obj = json.loads(line.strip())
                text = ' '.join(json_obj['content'])
                data.append({
                    'text': text,
                    'language': json_obj['language'],
                    'url': json_obj['url'],
                    'title': json_obj['metadata']['title']
                })
        return data

    def encode_texts(self):
        encoded = []
        for item in self.data:
            tokens = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in item['text'].split()]
            if len(tokens) < self.seq_length + 1:
                tokens += [self.word2idx['<PAD>']] * (self.seq_length + 1 - len(tokens))
            encoded.append(tokens)
        return encoded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.word2idx is None:
            raise ValueError("word2idx is not initialized")
        
        encoded = self.encoded_texts[idx]
        start_idx = random.randint(0, len(encoded) - self.seq_length - 1)
        src = torch.tensor(encoded[start_idx:start_idx + self.seq_length])
        tgt = torch.tensor(encoded[start_idx + 1:start_idx + self.seq_length + 1])
        return src, tgt, self.data[idx]['language']

def generate_text(model, start_text, word2idx, idx2word, max_length=50, temperature=0.8):
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

def build_vocab(data, vocab_size):
    word_counts = Counter(word for item in data for word in item['text'].split())
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
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location='cpu')
        
        if isinstance(model, DistributedDataParallel):
            new_state_dict = {f"module.{k}": v for k, v in checkpoint['model_state_dict'].items()}
        else:
            new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        
        model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded. Resuming training from epoch {epoch}")
        return epoch, loss
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0, float('inf')

def main(local_rank, world_size):
    # Enhanced Hyperparameters
    vocab_size = 30000
    d_model = 768
    nhead = 12
    num_layers = 16
    batch_size = 32
    seq_length = 128
    num_epochs = 100
    learning_rate = 1e-4
    custom_dataset_path = 'LLM-test/content/dataset.jsonl'
    checkpoint_filename = 'checkpoint.pth'
    accumulation_steps = 8
    patience = 10
    warmup_steps = 4000

    # Set up distributed training if applicable
    if world_size > 1:
        is_distributed = True
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        is_distributed = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Is distributed: {is_distributed}")

    if not is_distributed or local_rank == 0:
        try:
            temp_dataset = JSONLDataset(custom_dataset_path, word2idx=None, seq_length=seq_length)
            word2idx, idx2word = build_vocab(temp_dataset.data, vocab_size)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return
    else:
        word2idx, idx2word = None, None

    # Broadcast data to all processes if in distributed mode
    if is_distributed:
        word2idx = [word2idx] if local_rank == 0 else [None]
        torch.distributed.broadcast_object_list(word2idx, src=0)
        word2idx = word2idx[0]

        idx2word = [idx2word] if local_rank == 0 else [None]
        torch.distributed.broadcast_object_list(idx2word, src=0)
        idx2word = idx2word[0]

    # Create dataset and dataloader
    dataset = JSONLDataset(custom_dataset_path, word2idx, seq_length)
    num_workers = min(multiprocessing.cpu_count(), 8)

    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=2)

    # Create the enhanced model
    model = EnhancedTransformer(vocab_size, d_model, nhead, num_layers, dim_feedforward=3072, dropout=0.2)
    model = model.to(device)
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'], label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
    
    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos((step - warmup_steps) / (num_epochs * len(dataloader) - warmup_steps) * math.pi)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
        
        for batch, (src, tgt, language) in enumerate(progress_bar):
            src, tgt = src.to(device), tgt.to(device)
            
            with autocast():
                output = model(src)
                loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            
            if (batch + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            total_loss += loss.item() * accumulation_steps
            
            if not is_distributed or local_rank == 0:
                progress_bar.set_postfix({'loss': loss.item() * accumulation_steps, 'lr': scheduler.get_last_lr()[0]})
        
        avg_loss = total_loss / len(dataloader)
        
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