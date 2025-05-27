# 随机搜索代码
import os, random, pickle
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
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
ids = [0, 1, 2]

def load_data(fp):
    with open(fp, 'r') as f:
        return ''.join(d.strip() + ' ' for d in f.read().splitlines()).strip()

def create_char_mapping(txt):
    c2i, i2c = {}, {}
    for ch in txt:
        if ch not in c2i:
            idx = len(c2i)
            c2i[ch], i2c[idx] = idx, ch
    return c2i, i2c

def build_sequences(txt, c2i, a):
    win = a.window
    xs = [[c2i[c] for c in txt[i:i+win]] for i in range(len(txt)-win)]
    ys = [c2i[txt[i+win]] for i in range(len(txt)-win)]
    a.sequences, a.targets = np.array(xs), np.array(ys)

def train(args, model):
    crit = nn.CrossEntropyLoss().cuda()
    opt = optim.Adam(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(TensorDataset(torch.tensor(args.sequences),
                                      torch.tensor(args.targets)),
                        batch_size=args.batch_size, shuffle=True)
    best, wait, pat = float('inf'), 0, 5
    model.train()
    losses = []
    for ep in range(args.num_epochs):
        tot = 0
        for xb, yb in loader:
            xb, yb = xb.cuda(), yb.cuda()
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        ep_loss = tot / len(loader)
        losses.append(ep_loss)
        print(f"Epoch {ep+1:3d}/{args.num_epochs}  Loss={ep_loss:.6f}")
        if ep_loss < best: best, wait = ep_loss, 0
        else: wait += 1
        if wait >= pat:
            print(f">> 早停于第 {ep+1} 轮"); break
    return np.mean(losses)

def main(file, bs, hd, win, lr, ep):
    args = Args()
    args.batch_size, args.hidden_dim = bs, hd
    args.window, args.learning_rate, args.num_epochs = win, lr, ep
    txt = load_data(file)
    c2i, i2c = create_char_mapping(txt)
    build_sequences(txt, c2i, args)
    model = DomainGenerator(args, len(c2i))
    model = nn.DataParallel(model, device_ids=ids).cuda()
    avg_loss = train(args, model)
    save_model(model, "model/dnschanger/rand_dnschanger_generator.pth")
    with open("pickle/dnschanger/rand_dnschanger_char_map.pkl", "wb") as f:
        pickle.dump((c2i, i2c, args), f)
    return avg_loss
# 随机搜索 
def optimize_hyperparameters():
    bounds = {
        'batch_size': (32, 64),
        'hidden_dim' : (1024, 2048),
        'window'     : (16, 20),
        'learning_rate': (1e-5, 1e-4),
        'num_epochs' : (50, 70)
    }
    n_iter = 30        # 随机搜索迭代次数
    best_cfg, best_loss = None, float('inf')
    for i in range(1, n_iter + 1):
        bs = random.randint(*bounds['batch_size'])
        hd = random.randint(*bounds['hidden_dim'])
        win = random.randint(*bounds['window'])
        lr = 10 ** random.uniform(np.log10(bounds['learning_rate'][0]),
                                  np.log10(bounds['learning_rate'][1]))
        ep = random.randint(*bounds['num_epochs'])
        print(f"\n=== 随机尝试 {i}/{n_iter}: bs={bs}, hd={hd}, win={win}, "
              f"lr={lr:.5g}, ep={ep} ===")
        loss = main("data/dnschanger/dnschanger.txt", bs, hd, win, lr, ep)
        if loss < best_loss:
            best_loss, best_cfg = loss, (bs, hd, win, lr, ep)
            print(f"<< 当前最佳: loss={best_loss:.6f} | cfg={best_cfg}")
    print("\n*** 随机搜索完成 ***")
    print(f"最优超参数: batch_size={best_cfg[0]}, hidden_dim={best_cfg[1]}, "
          f"window={best_cfg[2]}, learning_rate={best_cfg[3]:.5g}, "
          f"num_epochs={best_cfg[4]}  | 平均损失={best_loss:.6f}")

if __name__ == "__main__":
    optimize_hyperparameters()
