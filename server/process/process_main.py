#Code by ChatGPT(추후 검토 및 수정 필요. )

import re
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader

# 1. 텍스트 정제
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# 2. 문장 읽기
def read_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    return [sentence.strip() for sentence in sentences]

# 3. 토크나이제이션
def tokenize(text):
    return text.split()

# 4. 단어 집합 구축
def build_vocab(sentences, min_freq=1):
    counter = Counter()
    for sentence in sentences:
        counter.update(sentence)
    vocab = {word: idx for idx, (word, freq) in enumerate(counter.items()) if freq >= min_freq}
    vocab['<pad>'] = len(vocab)
    vocab['<unk>'] = len(vocab)
    return vocab

# 5. 인덱스 변환
def text_to_indices(tokens, vocab):
    return [vocab.get(token, vocab['<unk>']) for token in tokens]

# 6. 패딩
def pad_sequences(sequences, max_len, pad_token):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_len:
            padded_seq = seq + [pad_token] * (max_len - len(seq))
        else:
            padded_seq = seq[:max_len]
        padded_sequences.append(padded_seq)
    return padded_sequences

# 7. 데이터셋 정의
class CorpusDataset(Dataset):
    def __init__(self, sentences, vocab, max_len):
        self.sentences = sentences
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        padded_sentence = pad_sequences([sentence], self.max_len, self.vocab['<pad>'])[0]
        return torch.tensor(padded_sentence)

corpus_file_path = 'corpus.txt'
sentences = read_corpus(corpus_file_path)
preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]
tokenized_sentences = [tokenize(sentence) for sentence in preprocessed_sentences]
vocab = build_vocab(tokenized_sentences)
indexed_sentences = [text_to_indices(sentence, vocab) for sentence in tokenized_sentences]
max_len = max(len(seq) for seq in indexed_sentences)
pad_token = vocab['<pad>']
padded_sequences = pad_sequences(indexed_sentences, max_len, pad_token)
dataset = CorpusDataset(padded_sequences, vocab, max_len)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 데이터 확인
for batch in dataloader:
    print(batch)
    break