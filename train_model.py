import torch
import torch.nn as nn
import torch.optim as optim
import random
from data_preprocessing import main


# 定义模型
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds)
        log_probs = torch.log_softmax(out, dim=1)
        return log_probs


# 训练模型
def train_model(model, training_data, num_epochs=100, batch_size=10):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(num_epochs):
        total_loss = 0
        random.shuffle(training_data)

        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i + batch_size]
            center_words, context_words = zip(*batch)
            center_words_tensor = torch.tensor(center_words, dtype=torch.long)
            context_words_tensor = torch.tensor(context_words, dtype=torch.long)

            model.zero_grad()
            log_probs = model(center_words_tensor)
            loss = criterion(log_probs, context_words_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Loss: {total_loss:.4f}')


# 主函数
def main():
    try:
        # 加载预处理的数据
        vocab_size, word_to_idx, idx_to_word, training_data = torch.load('preprocessed_data.pth', weights_only=True)

        # 参数设置
        embedding_dim = 10
        model = SkipGramModel(vocab_size, embedding_dim)

        # 训练模型
        train_model(model, training_data)

        # 保存模型
        torch.save(model.state_dict(), 'skip_gram_model.pth')
        print("Model training completed successfully.")
    except Exception as e:
        print(f"An error occurred during model training: {e}")


if __name__ == '__main__':
    main()