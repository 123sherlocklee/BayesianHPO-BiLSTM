import os, random, itertools, pickle
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from bi_lstm import DomainGenerator, save_model   

class Args:
    def __init__(self):
        self.batch_size = 64
        self.hidden_dim = 2048
        self.window = 15
        self.learning_rate = 1e-5
        self.num_epochs = 100
        self.sequences, self.targets = [], []

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
ids = [0, 1, 2]

def load_data(file_path):
    with open(file_path, 'r') as f:
        domains = f.read().strip().splitlines()
    return ''.join(d.strip() + ' ' for d in domains).strip()

def create_char_mapping(text):
    char2idx, idx2char = {}, {}
    for ch in text:
        if ch not in char2idx:
            idx = len(char2idx)
            char2idx[ch], idx2char[idx] = idx, ch
    return char2idx, idx2char

def build_sequences(text, char2idx, args):
    win = args.window
    xs, ys = [], []
    for i in range(len(text) - win):
        xs.append([char2idx[c] for c in text[i:i+win]])
        ys.append(char2idx[text[i+win]])
    args.sequences, args.targets = np.array(xs), np.array(ys)

def train(args, model):
    history, criterion = {'tr_loss': []}, nn.CrossEntropyLoss().cuda()
    optim_ = optim.Adam(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(TensorDataset(torch.tensor(args.sequences), 
                                      torch.tensor(args.targets)),
                        batch_size=args.batch_size, shuffle=True)

    patience, best_loss, counter = 5, float('inf'), 0
    model.train()
    for ep in range(args.num_epochs):
        loss_sum = 0
        for xb, yb in loader:
            xb, yb = xb.cuda(), yb.cuda()
            loss = criterion(model(xb), yb)
            optim_.zero_grad(); loss.backward(); optim_.step()
            loss_sum += loss.item()
        ep_loss = loss_sum / len(loader)
        history['tr_loss'].append(ep_loss)
        print(f"Epoch {ep+1:3d}/{args.num_epochs}  Loss={ep_loss:.6f}")
        if ep_loss < best_loss: best_loss, counter = ep_loss, 0
        else: counter += 1
        if counter >= patience:
            print(f">> 早停于第 {ep+1} 轮"); break
    return np.mean(history['tr_loss'])

def main(input_file, batch_size, hidden_dim, window, lr, epochs):
    args = Args()
    args.batch_size, args.hidden_dim = batch_size, hidden_dim
    args.window, args.learning_rate, args.num_epochs = window, lr, epochs
    text = load_data(input_file)
    char2idx, idx2char = create_char_mapping(text)
    build_sequences(text, char2idx, args)

    vocab = len(char2idx)
    model = DomainGenerator(args, vocab)
    model = nn.DataParallel(model, device_ids=ids).cuda()
    avg_loss = train(args, model)

    save_model(model, "model/dnschanger/grid_dnschanger_generator.pth")
    with open("pickle/dnschanger/grid_dnschanger_char_map.pkl", "wb") as f:
        pickle.dump((char2idx, idx2char, args), f)
    return avg_loss
    
# 网格搜索 
def optimize_hyperparameters():
    # 手动列出每个超参数候选值
    grid = {
        'batch_size': [32, 64],
        'hidden_dim': [1024, 1536, 2048],
        'window': [16, 18, 20],
        'learning_rate': [1e-5, 5e-5, 1e-4],
        'num_epochs': [50, 60, 70]
    }
    combos = list(itertools.product(*grid.values()))
    print(f">> 网格搜索组合总数: {len(combos)}")
    best_cfg, best_loss = None, float('inf')
    for idx, (bs, hd, win, lr, ep) in enumerate(combos, 1):
        print(f"\n=== 组合 {idx}/{len(combos)}: "
              f"bs={bs}, hd={hd}, win={win}, lr={lr:.5g}, ep={ep} ===")
        loss = main("data/dnschanger/dnschanger.txt",
                    bs, hd, win, lr, ep)
        if loss < best_loss:
            best_loss, best_cfg = loss, (bs, hd, win, lr, ep)
            print(f"<< 当前最佳: loss={best_loss:.6f} | cfg={best_cfg}")
    print("\n*** 网格搜索完成 ***")
    print(f"最优超参数: batch_size={best_cfg[0]}, hidden_dim={best_cfg[1]}, "
          f"window={best_cfg[2]}, learning_rate={best_cfg[3]:.5g}, "
          f"num_epochs={best_cfg[4]}  | 平均损失={best_loss:.6f}")

if __name__ == "__main__":
    optimize_hyperparameters()
