import torch
from collections import Counter
import random

# 读取文本数据
def read_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        sentences = [sentence.strip().lower().split() for sentence in text.split('\n') if sentence.strip()]
        return sentences
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the file: {e}")

# 构建词汇表
def build_vocab(sentences):
    all_words = [word for sentence in sentences for word in sentence]
    vocab = Counter(all_words)
    vocab_size = len(vocab)
    word_to_idx = {word: idx for idx, word in enumerate(vocab.keys())}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return vocab_size, word_to_idx, idx_to_word

# 创建训练数据
def create_training_data(sentences, word_to_idx, window_size=2):
    training_data = []
    for sentence in sentences:
        for i, word in enumerate(sentence):
            center_word = word_to_idx[word]
            for j in range(max(0, i - window_size), min(i + window_size + 1, len(sentence))):
                if j != i:
                    context_word = word_to_idx[sentence[j]]
                    training_data.append((center_word, context_word))
    return training_data

# 主函数
def main():
    file_path = 'dataset.txt'
    sentences = read_data(file_path)
    vocab_size, word_to_idx, idx_to_word = build_vocab(sentences)
    training_data = create_training_data(sentences, word_to_idx)
    return vocab_size, word_to_idx, idx_to_word, training_data

if __name__ == '__main__':
    try:
        vocab_size, word_to_idx, idx_to_word, training_data = main()
        # 保存词汇表和训练数据
        torch.save((vocab_size, word_to_idx, idx_to_word, training_data), 'preprocessed_data.pth')
        print("Data preprocessing completed successfully.")
    except Exception as e:
        print(f"An error occurred during data preprocessing: {e}")