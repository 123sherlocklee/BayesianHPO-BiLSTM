import os
import torch
import torch.nn as nn
import numpy as np
from bi_lstm import DomainGenerator, load_model
import pickle
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
ids = [0,1,2,3]

# 超参数
class Args:
    def __init__(self):
        self.batch_size = 16
        self.hidden_dim = 1024
        self.window = 16
        self.learning_rate = 0.0001
        self.num_epochs = 70
        self.sequences = list()
        self.targets = list()

def is_valid_domain(domain):
    domain = domain.rstrip()  # 去除域名末尾的空格
    pattern = r'^[a-z0-9]+(-[a-z0-9]+)*\.[a-z]{2,4}$'
    return re.match(pattern, domain) is not None

# 生成新域名
def generate_domain(model, sequences, idx2char, n_chars):
    model.eval()
    # 定义softmax函数
    softmax = nn.Softmax(dim=1).cuda()
    # 给定的idx来定义模式
    #sequences= sequences.reshape((sequences.shape[0], sequences.shape[1], 1))
    pattern = sequences[-1]
    
    # 利用字典，它输出了Pattern
    print("\nPattern: \n")
    print(''.join([idx2char[value] for value in pattern]), "\"")
    
    # 在full_prediction中，我们将保存完整的预测
    full_prediction = pattern.copy()
    
    # 预测开始，它将被预测为一个给定的字符长度
    for i in range(n_chars):
        # 转换为tensor
        pattern = torch.tensor(pattern, dtype=torch.long).cuda()
        pattern = pattern.view(1,-1)
        # 预测
        prediction = model(pattern)
        # 将softmax函数应用于预测张量
        prediction = softmax(prediction)
        # 预测张量被转换成一个numpy数组
        prediction = prediction.squeeze().detach().cpu().numpy()
        # 取概率最大的idx
        arg_max = np.argmax(prediction)
        # 将当前张量转换为numpy数组
        pattern = pattern.squeeze().detach().cpu().numpy()
        # 窗口向右1个字符
        pattern = pattern[1:]
        # 新pattern是由“旧”pattern+预测的字符组成的
        pattern = np.append(pattern, arg_max)
        # 保存完整的预测
        full_prediction = np.append(full_prediction, arg_max)
        
    print("Prediction: \n")
    print(''.join([idx2char[value] for value in full_prediction]), "\"")
    return ''.join([idx2char[value] for value in full_prediction])
    
def main(model_path, n_chars):
    with open("pickle/bazarbackdoor/examplev2_char_mappings_2.pkl", 'rb') as f:
        char2idx, idx2char, args= pickle.load(f)
    vocab_size = len(char2idx)

    model = DomainGenerator(args, vocab_size)
    model = nn.DataParallel(model, device_ids=ids).cuda()
    model = load_model(model, model_path)

    #sequences = "biictowy.bazar  "
    #sequences = [char2idx[char] for char in sequences]
    
    generated_text  = generate_domain(model, args.sequences, idx2char, n_chars)
    generated_domains = generated_text.split(' ')

    with open("data/gen/examplev2_generator_2.txt", 'w') as f:
        for domain in generated_domains:
            f.write(domain + '\n')
    
if __name__ == "__main__":
    main("model/bazarbackdoor/examplev2_generator_2.pth", 3000)
