import torch
from data_preprocessing import main
from train_model import SkipGramModel


# 获取单词嵌入
def get_word_embedding(word, model, word_to_idx):
    word_idx = word_to_idx.get(word)
    if word_idx is None:
        raise ValueError(f"Word '{word}' not found in vocabulary.")
    with torch.no_grad():
        embed = model.embeddings(torch.tensor([word_idx]))
    return embed.numpy()


# 主函数
def main():
    try:
        # 加载预处理的数据
        vocab_size, word_to_idx, idx_to_word, _ = torch.load('preprocessed_data.pth', weights_only=True)

        # 参数设置
        embedding_dim = 10
        model = SkipGramModel(vocab_size, embedding_dim)
        model.load_state_dict(torch.load('skip_gram_model.pth', weights_only=True))
        model.eval()

        # 测试模型
        print(get_word_embedding('programming', model, word_to_idx))
        print(get_word_embedding('learning', model, word_to_idx))
        print("Model testing completed successfully.")
    except Exception as e:
        print(f"An error occurred during model testing: {e}")


if __name__ == '__main__':
    main()