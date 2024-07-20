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
from nltk.corpus import gutenberg
import multiprocessing
import sys
import argparse


# Download NLTK data
nltk.download('gutenberg', quiet=True)

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4, dropout=0.1, batch_first=True)
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

class TextDataset(Dataset):
    def __init__(self, texts, word2idx, seq_length):
        self.texts = texts
        self.word2idx = word2idx
        self.seq_length = seq_length
        
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

def generate_dataset(num_sentences):
    print(f"Generating dataset with {num_sentences} sentences...")
    texts = []
    for fileid in tqdm(gutenberg.fileids(), desc="Processing Gutenberg texts"):
        words = gutenberg.words(fileid)
        sentences = [' '.join(words[i:i+15]).lower() for i in range(0, len(words), 15)]
        texts.extend(sentences[:num_sentences // len(gutenberg.fileids())])
        if len(texts) >= num_sentences:
            break
    return texts[:num_sentences]

def build_vocab(texts, vocab_size):
    word_counts = Counter(word for text in texts for word in text.lower().split())
    vocab = ['<PAD>', '<UNK>'] + [word for word, _ in word_counts.most_common(vocab_size - 2)]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

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

# Base sentences
base_sentences = [
    "The quick brown fox jumps over the lazy dog.",
"A journey of a thousand miles begins with a single step.",
"To be or not to be, that is the question.",
"All that glitters is not gold.",
"Where there is a will, there is a way.",
"Actions speak louder than words.",
"Knowledge is power.",
"Practice makes perfect.",
"Time is money.",
"Better late than never.",
"Every cloud has a silver lining.",
"Two wrongs don't make a right.",
"When in Rome, do as the Romans do.",
"The early bird catches the worm.",
"Honesty is the best policy.",
"Laughter is the best medicine.",
"Necessity is the mother of invention.",
"A picture is worth a thousand words.",
"There's no place like home.",
"The pen is mightier than the sword.",
"Rome wasn't built in a day.",
"Don't count your chickens before they hatch.",
"A watched pot never boils.",
"All's fair in love and war.",
"Beauty is in the eye of the beholder.",
"Actions speak louder than words.",
"Fortune favors the bold.",
"Every rose has its thorn.",
"Absence makes the heart grow fonder.",
"Good things come to those who wait.",
"Don't put all your eggs in one basket.",
"Honesty is the best policy.",
"Don't bite the hand that feeds you.",
"Jack of all trades, master of none.",
"Birds of a feather flock together.",
"Too many cooks spoil the broth.",
"Strike while the iron is hot.",
"A stitch in time saves nine.",
"Beggars can't be choosers.",
"Curiosity killed the cat.",
"Don't judge a book by its cover.",
"Fortune favors the brave.",
"Hit the nail on the head.",
"Let the cat out of the bag.",
"Make hay while the sun shines.",
"People who live in glass houses shouldn't throw stones.",
"A penny saved is a penny earned.",
"Practice what you preach.",
"Rome wasn't built in a day.",
"An apple a day keeps the doctor away.",
"Too much of a good thing is bad.",
"Don't cry over spilled milk.",
"Time flies when you're having fun.",
"Where there's smoke, there's fire.",
"Better safe than sorry.",
"Fools rush in where angels fear to tread.",
"A chain is only as strong as its weakest link.",
"All is fair in love and war.",
"Don't bite off more than you can chew.",
"Actions speak louder than words.",
"Great minds think alike.",
"Too many cooks spoil the broth.",
"A watched pot never boils.",
"The grass is always greener on the other side.",
"Don't count your chickens before they hatch.",
"Every rose has its thorn.",
"Absence makes the heart grow fonder.",
"Better late than never.",
"Don't judge a book by its cover.",
"Good things come to those who wait.",
"Curiosity killed the cat.",
"Strike while the iron is hot.",
"Where there's a will, there's a way.",
"A penny for your thoughts.",
"Don't put all your eggs in one basket.",
"Jack of all trades, master of none.",
"People who live in glass houses shouldn't throw stones.",
"Let sleeping dogs lie.",
"Actions speak louder than words.",
"Two heads are better than one.",
"Don't put the cart before the horse.",
"Make hay while the sun shines.",
"Birds of a feather flock together.",
"Strike while the iron is hot.",
"A stitch in time saves nine.",
"Time and tide wait for no man.",
"Every cloud has a silver lining.",
"Necessity is the mother of invention.",
"Practice makes perfect.",
"Fortune favors the bold.",
"All good things must come to an end.",
"Every dog has its day.",
"Don't put all your eggs in one basket.",
"Actions speak louder than words.",
"Better safe than sorry.",
"Don't cry over spilled milk.",
"Two wrongs don't make a right.",
"Beauty is in the eye of the beholder.",
"Fortune favors the brave.",
"Good things come to those who wait.",
"Strike while the iron is hot.",
"Time is money.",
"An apple a day keeps the doctor away.",
"Don't judge a book by its cover.",
"Curiosity killed the cat.",
"Rome wasn't built in a day.",
"A chain is only as strong as its weakest link.",
"Jack of all trades, master of none.",
"Every rose has its thorn.",
"Practice what you preach.",
"Too many cooks spoil the broth.",
"A penny saved is a penny earned.",
"Let the cat out of the bag.",
"Too much of a good thing is bad.",
"Birds of a feather flock together.",
"Where there's a will, there's a way.",
"Fortune favors the bold.",
"Make hay while the sun shines.",
"A watched pot never boils.",
"Better late than never.",
"Don't bite the hand that feeds you.",
"Actions speak louder than words.",
"Every cloud has a silver lining.",
"Don't judge a book by its cover.",
"Curiosity killed the cat.",
"Strike while the iron is hot.",
"An apple a day keeps the doctor away.",
"Good things come to those who wait.",
"Time and tide wait for no man.",
"Jack of all trades, master of none.",
"Practice makes perfect.",
"Birds of a feather flock together.",
"A chain is only as strong as its weakest link.",
"Better safe than sorry.",
"Don't put all your eggs in one basket.",
"Fortune favors the brave.",
"A penny for your thoughts.",
"Too many cooks spoil the broth.",
"Make hay while the sun shines.",
"Every dog has its day.",
"Strike while the iron is hot.",
"Actions speak louder than words.",
"A stitch in time saves nine.",
"Jack of all trades, master of none.",
"Fortune favors the bold.",
"Practice what you preach.",
"Time is money.",
"Too much of a good thing is bad.",
"Don't count your chickens before they hatch.",
"Let sleeping dogs lie.",
"A watched pot never boils.",
"Beauty is in the eye of the beholder.",
"Every rose has its thorn.",
"Birds of a feather flock together.",
"A penny saved is a penny earned.",
"Don't cry over spilled milk.",
"Jack of all trades, master of none.",
"Too many cooks spoil the broth.",
"A picture is worth a thousand words.",
"Where there's smoke, there's fire.",
"Fortune favors the brave.",
"Curiosity killed the cat.",
"Make hay while the sun shines.",
"Good things come to those who wait.",
"Actions speak louder than words.",
"Every cloud has a silver lining.",
"Don't judge a book by its cover.",
"Better late than never.",
"Rome wasn't built in a day.",
"Strike while the iron is hot.",
"An apple a day keeps the doctor away.",
"A watched pot never boils.",
"Birds of a feather flock together.",
"Don't bite the hand that feeds you.",
"Fortune favors the bold.",
"Time is money.",
"Practice makes perfect.",
"A chain is only as strong as its weakest link.",
"Every dog has its day.",
"Too much of a good thing is bad.",
"Make hay while the sun shines.",
"Too many cooks spoil the broth.",
"Don't put all your eggs in one basket.",
"Better safe than sorry.",
"Strike while the iron is hot.",
"Good things come to those who wait.",
"Don't cry over spilled milk.",
"Beauty is in the eye of the beholder.",
"Actions speak louder than words.",
"A penny saved is a penny earned.",
"Every rose has its thorn.",
"Curiosity killed the cat.",
"Time flies when you're having fun.",
"An apple a day keeps the doctor away.",
"Practice makes perfect.",
"Jack of all trades, master of none.",
"Don't put the cart before the horse.",
"Strike while the iron is hot.",
"Make hay while the sun shines.",
"A watched pot never boils.",
"Too many cooks spoil the broth.",
"Fortune favors the brave.",
"Birds of a feather flock together.",
"Every cloud has a silver lining.",
"Don't bite the hand that feeds you.",
"Better late than never.",
"Too much of a good thing is bad.",
"Good things come to those who wait.",
"A penny saved is a penny earned.",
"Curiosity killed the cat.",
"Practice what you preach.",
"Time flies when you're having fun.",
"Make hay while the sun shines.",
"Jack of all trades, master of none.",
"Every dog has its day.",
"Fortune favors the bold.",
"Don't judge a book by its cover.",
"A chain is only as strong as its weakest link.",
"Birds of a feather flock together.",
"Strike while the iron is hot.",
"Too many cooks spoil the broth.",
"An apple a day keeps the doctor away.",
"Practice makes perfect.",
"Every rose has its thorn.",
"Don't cry over spilled milk.",
"Beauty is in the eye of the beholder.",
"Good things come to those who wait.",
"Jack of all trades, master of none.",
"Better late than never.",
"Actions speak louder than words.",
"Curiosity killed the cat.",
"A watched pot never boils.",
"Fortune favors the bold.",
"Too many cooks spoil the broth.",
"Make hay while the sun shines.",
"Every cloud has a silver lining.",
"Don't judge a book by its cover.",
"An apple a day keeps the doctor away.",
"Time flies when you're having fun.",
"Strike while the iron is hot.",
"Too much of a good thing is bad.",
"A penny saved is a penny earned.",
"Practice makes perfect.",
"Birds of a feather flock together.",
"Good things come to those who wait.",
"Better safe than sorry.",
"Every dog has its day.",
"Jack of all trades, master of none.",
"Fortune favors the brave.",
"A chain is only as strong as its weakest link.",
"Make hay while the sun shines.",
"Curiosity killed the cat.",
"Don't cry over spilled milk.",
"Beauty is in the eye of the beholder.",
"Time and tide wait for no man.",
"An apple a day keeps the doctor away.",
"Strike while the iron is hot.",
"Too many cooks spoil the broth.",
"Actions speak louder than words.",
"Every rose has its thorn.",
"Jack of all trades, master of none.",
"Don't put all your eggs in one basket.",
"Better late than never.",
"Good things come to those who wait.",
"Birds of a feather flock together.",
"Make hay while the sun shines.",
"Fortune favors the bold.",
"Curiosity killed the cat.",
"Practice makes perfect.",
"Every cloud has a silver lining.",
"Too many cooks spoil the broth.",
"A watched pot never boils.",
"Too much of a good thing is bad.",
"Jack of all trades, master of none.",
"Time flies when you're having fun.",
"Don't cry over spilled milk.",
"Better safe than sorry.",
"Strike while the iron is hot.",
"Good things come to those who wait.",
"Birds of a feather flock together.",
"Every dog has its day.",
"Fortune favors the brave.",
"A penny saved is a penny earned.",
"An apple a day keeps the doctor away.",
"Don't bite the hand that feeds you.",
"Practice makes perfect.",
"Make hay while the sun shines.",
"Too many cooks spoil the broth.",
"Curiosity killed the cat.",
"Better late than never.",
"Jack of all trades, master of none.",
"Actions speak louder than words.",
"Every rose has its thorn.",
"Don't judge a book by its cover.",
"Strike while the iron is hot.",
"Good things come to those who wait.",
"Birds of a feather flock together.",
"Too much of a good thing is bad.",
"A chain is only as strong as its weakest link.",
"Fortune favors the brave.",
"Practice what you preach.",
"Every cloud has a silver lining.",
"A watched pot never boils.",
"Curiosity killed the cat.",
"Don't cry over spilled milk.",
"Better safe than sorry.",
"Time and tide wait for no man.",
"Jack of all trades, master of none.",
"Good things come to those who wait.",
"Too many cooks spoil the broth.",
"Fortune favors the bold.",
"Make hay while the sun shines.",
"An apple a day keeps the doctor away.",
"Actions speak louder than words.",
"Every rose has its thorn.",
"Birds of a feather flock together.",
"A chain is only as strong as its weakest link.",
"Better late than never.",
"Too much of a good thing is bad.",
"Curiosity killed the cat.",
"Practice makes perfect.",
"Good things come to those who wait.",
"Time flies when you're having fun.",
"Strike while the iron is hot.",
"A watched pot never boils.",
"Fortune favors the bold.",
"Every dog has its day.",
"Jack of all trades, master of none.",
"Too many cooks spoil the broth.",
"Don't judge a book by its cover.",
"An apple a day keeps the doctor away.",
"Actions speak louder than words.",
"Every cloud has a silver lining.",
"Make hay while the sun shines.",
"Better safe than sorry.",
"Curiosity killed the cat.",
"Good things come to those who wait.",
"A chain is only as strong as its weakest link.",
"Birds of a feather flock together.",
"Too many cooks spoil the broth.",
"Don't cry over spilled milk.",
"Time flies when you're having fun.",
"Practice makes perfect.",
"Fortune favors the brave.",
"Every rose has its thorn.",
"Better late than never.",
"An apple a day keeps the doctor away.",
"Make hay while the sun shines.",
"Actions speak louder than words.",
"Too much of a good thing is bad.",
"Don't judge a book by its cover.",
"Jack of all trades, master of none.",
"Curiosity killed the cat.",
"Birds of a feather flock together.",
"A watched pot never boils.",
"Good things come to those who wait.",
"Better late than never.",
"Every cloud has a silver lining.",
"Strike while the iron is hot.",
"Time flies when you're having fun.",
"Fortune favors the bold.",
"Too many cooks spoil the broth.",
"An apple a day keeps the doctor away.",
"Don't cry over spilled milk.",
"Practice makes perfect.",
"Birds of a feather flock together.",
"Good things come to those who wait.",
"Better safe than sorry.",
"Every rose has its thorn.",
"Too much of a good thing is bad.",
"Curiosity killed the cat.",
"Make hay while the sun shines.",
"Every dog has its day.",
"Fortune favors the brave.",
"Time and tide wait for no man.",
"Jack of all trades, master of none.",
"Actions speak louder than words.",
"A chain is only as strong as its weakest link.",
"Too many cooks spoil the broth.",
"An apple a day keeps the doctor away.",
"Practice makes perfect.",
"Don't bite the hand that feeds you.",
"Birds of a feather flock together.",
"Good things come to those who wait.",
"Curiosity killed the cat.",
"Better late than never.",
"Strike while the iron is hot.",
"Every cloud has a silver lining.",
"Too much of a good thing is bad.",
"Make hay while the sun shines.",
"Jack of all trades, master of none.",
"Fortune favors the brave.",
"Don't cry over spilled milk.",
"Practice makes perfect.",
"Every rose has its thorn.",
"A watched pot never boils.",
"Too many cooks spoil the broth.",
"Good things come to those who wait.",
"An apple a day keeps the doctor away.",
"Actions speak louder than words.",
"Curiosity killed the cat.",
"Birds of a feather flock together.",
"Better late than never.",
"Every dog has its day.",
"Fortune favors the brave.",
"Make hay while the sun shines.",
"Strike while the iron is hot.",
"Too much of a good thing is bad.",
"Don't judge a book by its cover.",
"A chain is only as strong as its weakest link.",
"Good things come to those who wait.",
"Curiosity killed the cat.",
"Practice makes perfect.",
"Every cloud has a silver lining.",
"Birds of a feather flock together.",
"Too many cooks spoil the broth.",
"An apple a day keeps the doctor away.",
"Fortune favors the brave.",
"Don't cry over spilled milk.",
"Actions speak louder than words.",
"Better safe than sorry.",
"Time flies when you're having fun.",
"Jack of all trades, master of none.",
"Make hay while the sun shines.",
"Every rose has its thorn.",
"A watched pot never boils.",
"Too much of a good thing is bad.",
"Curiosity killed the cat.",
"Good things come to those who wait.",
"Strike while the iron is hot.",
"Every dog has its day.",
"Fortune favors the bold.",
"Birds of a feather flock together.",
"Better late than never.",
"Actions speak louder than words.",
"Don't bite the hand that feeds you.",
"Too many cooks spoil the broth.",
"Practice makes perfect.",
"An apple a day keeps the doctor away.",
"Make hay while the sun shines.",
"Curiosity killed the cat.",
"Every cloud has a silver lining.",
"Strike while the iron is hot.",
"Jack of all trades, master of none.",
"Fortune favors the brave.",
"Good things come to those who wait.",
"Too many cooks spoil the broth.",
"Better safe than sorry.",
"Birds of a feather flock together.",
"Time flies when you're having fun.",
"Every rose has its thorn.",
"Curiosity killed the cat.",
"Don't cry over spilled milk.",
"Make hay while the sun shines.",
"A watched pot never boils.",
"Practice makes perfect.",
"Good things come to those who wait.",
"Too much of a good thing is bad.",
"Strike while the iron is hot.",
"Fortune favors the brave.",
"Birds of a feather flock together.",
"Every dog has its day.",
"Better late than never.",
"An apple a day keeps the doctor away.",
"Curiosity killed the cat.",
"Too many cooks spoil the broth.",
"Good things come to those who wait.",
"Practice makes perfect.",
"Don't judge a book by its cover.",
"Every cloud has a silver lining.",
"Strike while the iron is hot.",
"Fortune favors the bold.",
"Make hay while the sun shines.",
"Time flies when you're having fun.",
"Jack of all trades, master of none.",
"Every rose has its thorn.",
"Too much of a good thing is bad.",
"Curiosity killed the cat.",
"Better safe than sorry.",
"Don't cry over spilled milk.",
"Good things come to those who wait.",
"Every dog has its day.",
"Strike while the iron is hot.",
"Birds of a feather flock together.",
"Too many cooks spoil the broth.",
"Practice makes perfect.",
"A watched pot never boils.",
"Fortune favors the brave.",
"Better late than never.",
"Make hay while the sun shines.",
"Good things come to those who wait.",
"Every cloud has a silver lining.",
"An apple a day keeps the doctor away.",
"Curiosity killed the cat.",
"Too many cooks spoil the broth.",
"Strike while the iron is hot.",
"Don't cry over spilled milk.",
"Better safe than sorry.",
"Every rose has its thorn.",
"Good things come to those who wait.",
"A chain is only as strong as its weakest link.",
"Practice makes perfect.",
"Every dog has its day.",
"Too much of a good thing is bad.",
"Make hay while the sun shines.",
"Don't judge a book by its cover.",
"Fortune favors the brave.",
"Birds of a feather flock together.",
"Curiosity killed the cat.",
"The quick brown fox jumps over the lazy dog.",
"A journey of a thousand miles begins with a single step.",
"To be or not to be, that is the question.",
"All that glitters is not gold.",
"Where there is a will, there is a way.",
"Actions speak louder than words.",
"Knowledge is power.",
"Practice makes perfect.",
"Time is money.",
"Better late than never.",
"Every cloud has a silver lining.",
"Two wrongs don't make a right.",
"When in Rome, do as the Romans do.",
"The early bird catches the worm.",
"Honesty is the best policy.",
"Laughter is the best medicine.",
"Necessity is the mother of invention.",
"A picture is worth a thousand words.",
"There's no place like home.",
"The pen is mightier than the sword.",
"Rome wasn't built in a day, and neither are the foundations of great achievements established overnight. It takes a blend of perseverance, dedication, and the unwavering commitment to the vision that guides you towards your goals, no matter how ambitious they may be.",
"Success often comes to those who are too busy to be looking for it. While some wait for opportunities to knock, others are constantly working, honing their skills, and preparing themselves for when those opportunities arise. It is through relentless effort and strategic planning that greatness is often achieved.",
"The greatest glory in living lies not in never falling, but in rising every time we fall. Every setback, every challenge, every obstacle faced is an opportunity to grow stronger, to learn, and to emerge more resilient than before, embracing the journey of continuous improvement.",
"In the end, it's not the years in your life that count. It's the life in your years. What we do with the time given to us, the experiences we gather, and the relationships we nurture define the essence of our existence far more than the mere passage of time.",
"Success is not final, failure is not fatal: It is the courage to continue that counts. The path to success is fraught with challenges and setbacks, but the true measure of achievement lies in one's ability to persist through adversity and to keep moving forward despite the obstacles encountered.",
"Life is what happens when you're busy making other plans. Often, we get so caught up in meticulously planning our future that we forget to appreciate and fully engage in the present moments that make up our everyday lives. It's these spontaneous and unplanned experiences that often bring the most joy.",
"The only limit to our realization of tomorrow is our doubts of today. Beliefs and doubts can shape our future more than any external factors. By overcoming our own reservations and embracing a mindset of possibility, we open doors to new opportunities and achieve our fullest potential.",
"It is not the strongest of the species that survive, nor the most intelligent, but the one most responsive to change. Adaptability and the ability to embrace change are crucial for survival and success in a world that is constantly evolving and presenting new challenges.",
"To accomplish great things, we must not only act, but also dream; not only plan, but also believe. Success requires a harmonious blend of imagination, action, and conviction. By dreaming big, planning meticulously, and believing in our potential, we set ourselves up for remarkable achievements.",
"Success usually comes to those who are too busy to be looking for it. It's often those who are deeply immersed in their work, passionately pursuing their goals, and committed to their craft who find success, as opposed to those who merely chase it.",
"Every great dream begins with a dreamer. Always remember, you have within you the strength, the patience, and the passion to reach for the stars to change the world. It is through our dreams and the perseverance to pursue them that we make meaningful impacts and drive innovation.",
"In three words I can sum up everything I've learned about life: it goes on. Life is a continuous journey with its share of ups and downs, successes and failures. Despite the challenges and setbacks, the essence of life is its relentless progression and the opportunity it offers for growth and renewal.",
"Life is really simple, but we insist on making it complicated. We often overcomplicate our lives with unnecessary worries, stresses, and complexities. By simplifying our approach and focusing on what truly matters, we can lead more fulfilling and balanced lives.",
"To live is the rarest thing in the world. Most people exist, that is all. Truly living involves engaging with life fully, embracing new experiences, pursuing passions, and making meaningful connections. It requires courage to step out of the ordinary and to seek a deeper, more vibrant existence.",
"Everything you’ve ever wanted is on the other side of fear. Fear is often the greatest barrier to achieving our goals and dreams. By confronting and overcoming our fears, we unlock the potential to attain what we truly desire and to experience life more fully.",
"The best way to predict your future is to create it. Rather than waiting for opportunities to come our way or for circumstances to change, we have the power to shape our own destiny through proactive efforts, deliberate actions, and visionary thinking.",
"Be not afraid of life. Believe that life is worth living, and your belief will help create the fact. A positive outlook and a belief in the value of life can profoundly influence our experiences and shape our reality, making life more meaningful and rewarding.",
"Success is stumbling from failure to failure with no loss of enthusiasm. The path to success is rarely smooth or straightforward. It is through persistence and maintaining a positive attitude in the face of setbacks that we ultimately achieve our goals and find success.",
"To succeed in life, you need two things: ignorance and confidence. Sometimes, a lack of awareness of potential obstacles combined with a strong self-belief can propel us forward and lead us to achieve remarkable things that might otherwise seem unattainable.",
"Life is a series of natural and spontaneous changes. Don’t resist them; that only creates sorrow. Let reality be reality. Let things flow naturally forward in whatever way they like. Embracing change and allowing things to unfold naturally often leads to more harmonious outcomes.",
"The only way to do great work is to love what you do. Passion and enthusiasm for one's work are essential ingredients for achieving excellence. When we are genuinely engaged and passionate about our endeavors, we are more likely to produce outstanding results and find personal fulfillment.",
"Every man is capable of greatness, but it takes the courage to pursue one's dreams and the determination to overcome obstacles. Greatness is not reserved for a select few but is achievable by anyone who is willing to put in the effort and face the challenges along the way.",
"Success does not consist in never making mistakes but in never making the same one a second time. Learning from our mistakes and using them as stepping stones rather than stumbling blocks is a key component of long-term success and personal growth.",
"Your time is limited, so don’t waste it living someone else’s life. Follow your own path and pursue your own passions. By focusing on what truly resonates with you, you create a life that is uniquely your own and more fulfilling.",
"Don't watch the clock; do what it does. Keep going. Time is a constant and unchangeable force, but how we use it is within our control. By staying focused and persistent, we make the most of the time available to us and progress towards our goals.",
"Every experience, no matter how bad it seems, holds within it a blessing of some kind. The key is to find it. Even in challenging situations, there are opportunities for learning, growth, and finding unexpected benefits that contribute to our overall journey.",
"Success is not how high you have climbed, but how you make a positive difference to the world. True success is measured not just by personal achievements but by the impact we have on others and the positive contributions we make to society.",
"Believe you can and you’re halfway there. Self-belief and confidence are critical factors in achieving success. When we have faith in our abilities and our potential, we are more likely to overcome obstacles and reach our goals.",
"Life isn’t about finding yourself. Life is about creating yourself. It is through our actions, choices, and creative endeavors that we shape who we are and become the individuals we aspire to be, rather than merely discovering an inherent self.",
"The best revenge is massive success. Rather than focusing on retaliation or harboring grudges, channeling energy into achieving great success can be the most effective way to demonstrate resilience and turn challenges into opportunities for triumph.",
"Your most unhappy customers are your greatest source of learning. Constructive feedback and criticism, although difficult to hear, provide invaluable insights that can help us improve and grow, leading to better products, services, and overall success.",
"The future belongs to those who believe in the beauty of their dreams. Vision and belief in one's aspirations are powerful driving forces that shape the future. By nurturing and pursuing our dreams with conviction, we create a future that reflects our deepest desires and goals.",
"Life is 10% what happens to us and 90% how we react to it. Our responses and attitudes towards the events and challenges we encounter play a significant role in determining the course of our lives and our overall well-being.",
"Success is to be measured not so much by the position that one has reached in life as by the obstacles which he has overcome. The true measure of success lies in overcoming challenges and persevering through difficulties to achieve one's goals.",
"To succeed, you need to find something to hold on to, something to motivate you, something to inspire you. Finding and focusing on sources of motivation and inspiration are crucial for driving oneself towards success and maintaining momentum.",
"Life is either a daring adventure or nothing at all. Embracing life with courage and a willingness to explore new experiences leads to a richer and more fulfilling existence, while avoiding risks often results in missed opportunities for growth and discovery.",
"Success is achieved and maintained by those who try and keep trying. Perseverance and a continuous effort to improve and overcome obstacles are key factors in achieving long-term success and reaching one's goals.",
"Don’t be pushed around by the fears in your mind. Be led by the dreams in your heart. While fears and doubts can be paralyzing, it is the pursuit of our dreams and passions that truly guides us towards meaningful and fulfilling accomplishments.",
"Opportunities don't happen. You create them. Rather than waiting for chances to come our way, we have the power to proactively seek out and create our own opportunities through initiative, creativity, and persistence.",
"Success is walking from failure to failure with no loss of enthusiasm. Maintaining a positive outlook and continuing to strive towards one's goals, even after encountering failures, is essential for achieving ultimate success and personal growth.",
"Life is what we make it, always has been, always will be. Our experiences and the quality of our lives are largely shaped by our actions, choices, and attitudes. By taking control and actively shaping our lives, we determine our own happiness and fulfillment."

]

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
    vocab_size = 5000
    d_model = 256
    nhead = 8
    num_layers = 6
    batch_size = 128
    seq_length = 30
    num_epochs = 1000
    learning_rate = 0.0001
    num_sentences = 10000
    checkpoint_filename = 'transformer_checkpoint.pth'
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

    # Generate dataset
    if not is_distributed or local_rank == 0:
        texts = generate_dataset(num_sentences)
        word2idx, idx2word = build_vocab(texts, vocab_size)
    else:
        texts, word2idx, idx2word = None, None, None

    if is_distributed:
        # Broadcast data to all processes
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
    num_workers = min(multiprocessing.cpu_count(), 8)
    dataset = TextDataset(texts, word2idx, seq_length)

    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=2)


    # Create the model
    model = SimpleTransformer(vocab_size, d_model, nhead, num_layers)
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


