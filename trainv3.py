import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from bi_lstm import DomainGenerator, save_model
import pickle
from torch.utils.data import DataLoader, TensorDataset
from bayes_opt import BayesianOptimization

# 超参数类
class Args:
    def __init__(self):
        self.batch_size = 64
        self.hidden_dim = 2048
        self.window = 15
        self.learning_rate = 0.00001
        self.num_epochs = 100
        self.sequences = list()
        self.targets = list()

# 设置随机种子
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# 设置可见的 GPU 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
ids = [0, 1, 2]

# 读取输入文件并提取域名
def load_data(file_path):
    with open(file_path, 'r') as f:
        domains = f.read().strip().splitlines()
    concatenated_domains = ''.join(domain.strip() + ' ' for domain in domains).strip()
    return concatenated_domains

# 将字符映射为数字，以用于模型输入
def create_char_mapping(domains):
    char2idx = dict()
    idx2char = dict()
    idx = 0
    for char in domains:
        if char not in char2idx:
            char2idx[char] = idx
            idx2char[idx] = char
            idx += 1
    return char2idx, idx2char

def build_sequences(text, char2idx, args):
    x = []
    y = []
    window = args.window
    
    for i in range(len(text) - window):
        sequence = text[i:i+window]
        sequence = [char2idx[char] for char in sequence]
        
        target = text[i+window]
        target = char2idx[target]
    
        x.append(sequence)
        y.append(target)
    
    x = np.array(x)
    y = np.array(y)
    
    args.sequences = x
    args.targets = y

def train(args, model):
    history = {'tr_loss': []}
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    dataset = TensorDataset(torch.tensor(args.sequences, dtype=torch.long), torch.tensor(args.targets, dtype=torch.long))
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    model.train()

    # 早停机制的参数
    patience = 5  # 允许的最大无改善epoch数
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.num_epochs):
        epoch_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataloader)
        history['tr_loss'].append(epoch_loss)
        print(f"Epoch: {epoch+1}, Loss: {epoch_loss:.5f}")

        # 早停逻辑
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return np.mean(history['tr_loss'])

def main(input_file, batch_size, hidden_dim, window, learning_rate, num_epochs):
    args = Args()
    args.batch_size = batch_size
    args.hidden_dim = hidden_dim
    args.window = window
    args.learning_rate = learning_rate
    args.num_epochs = num_epochs

    domains = load_data(input_file)
    char2idx, idx2char = create_char_mapping(domains)
    vocab_size = len(char2idx)
    
    build_sequences(domains, char2idx, args)

    model = DomainGenerator(args, vocab_size)
    model = nn.DataParallel(model, device_ids=ids).cuda()
    avg_loss = train(args, model)

    save_model(model, 'model/dnschanger/bys_dnschanger_generator_1.pth')
    
    with open('pickle/dnschanger/bys_dnschanger_char_mappings_1.pkl', 'wb') as f:
        pickle.dump((char2idx, idx2char, args), f)
        
    print(f'Model saved to model/dnschanger/bys_dnschanger_generator_1.pth')
    return -avg_loss  # 返回负的平均损失，因为贝叶斯优化是最大化目标函数

# 贝叶斯优化函数
def optimize_hyperparameters():
    def black_box_function(batch_size, hidden_dim, window, learning_rate, num_epochs):
        return main(
            "data/dnschanger/dnschanger.txt",
            int(batch_size),
            int(hidden_dim),
            int(window),
            learning_rate,
            int(num_epochs)
        )

    pbounds = {
        'batch_size': (32, 64),
        'hidden_dim': (1024, 2048),
        'window': (16, 20),
        'learning_rate': (0.00001, 0.0001),
        'num_epochs': (50, 70)
    }

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=seed,
    )

    optimizer.maximize(init_points=5, n_iter=25)

    print("Best parameters found: ", optimizer.max)

if __name__ == "__main__":
    optimize_hyperparameters()
