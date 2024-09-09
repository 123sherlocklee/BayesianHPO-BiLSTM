import os
import torch
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
ids = [0,1,2,3]

class DomainGenerator(nn.Module):
 
    def __init__(self, args, vocab_size):
        super(DomainGenerator, self).__init__()
    
        self.batch_size = args.batch_size
        self.hidden_dim = args.hidden_dim
        self.input_size = vocab_size
        self.num_classes = vocab_size
        self.sequence_len = args.window
    
        # Dropout
        #self.dropout = nn.Dropout(0.2)
    
        # Embedding 层
        self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
    
        # Bi-LSTM
        # 正向和反向
        self.lstm_cell_forward = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.lstm_cell_backward = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
    
        # LSTM 层
        self.lstm_cell = nn.LSTMCell(self.hidden_dim * 2, self.hidden_dim * 2)
    
        # Linear 层
        self.linear = nn.Linear(self.hidden_dim * 2, self.num_classes)
        
        # 初始化隐藏状态
        self.hs_forward = None
        self.cs_forward = None
        self.hs_backward = None
        self.cs_backward = None
        self.hs_lstm = None
        self.cs_lstm = None
        
    def init_hidden(self, batch_size):
        hs_forward = torch.zeros(batch_size, self.hidden_dim).cuda()
        cs_forward = torch.zeros(batch_size, self.hidden_dim).cuda()
        hs_backward = torch.zeros(batch_size, self.hidden_dim).cuda()
        cs_backward = torch.zeros(batch_size, self.hidden_dim).cuda()
        hs_lstm = torch.zeros(batch_size, self.hidden_dim * 2).cuda()
        cs_lstm = torch.zeros(batch_size, self.hidden_dim * 2).cuda()
        
        # 权重初始化
        torch.nn.init.kaiming_normal_(hs_forward)
        torch.nn.init.kaiming_normal_(cs_forward)
        torch.nn.init.kaiming_normal_(hs_backward)
        torch.nn.init.kaiming_normal_(cs_backward)
        torch.nn.init.kaiming_normal_(hs_lstm)
        torch.nn.init.kaiming_normal_(cs_lstm)
        
        return (hs_forward, cs_forward), (hs_backward, cs_backward), (hs_lstm, cs_lstm)
        
    def forward(self, x):
        batch_size = x.size(0)
        if self.hs_forward is None :
            (self.hs_forward, self.cs_forward), (self.hs_backward, self.cs_backward), (self.hs_lstm, self.cs_lstm) = self.init_hidden(batch_size)
        
        # 从 idx 到 embedding
        out = self.embedding(x)
        
        # 为LSTM准备shape
        out = out.view(self.sequence_len, batch_size, -1)
        
        forward = []
        backward = []
        
        # 解开Bi-LSTM
        # 正向
        for i in range(self.sequence_len):
            self.hs_forward, self.cs_forward = self.lstm_cell_forward(out[i], (self.hs_forward, self.cs_forward))
            #self.hs_forward = self.dropout(self.hs_forward)
            #self.cs_forward = self.dropout(self.cs_forward)
            forward.append(self.hs_forward)
            
        # 反向
        for i in reversed(range(self.sequence_len)):
            self.hs_backward, self.cs_backward = self.lstm_cell_backward(out[i], (self.hs_backward, self.cs_backward))
            #self.hs_backward = self.dropout(self.hs_backward)
            #self.cs_backward = self.dropout(self.cs_backward)
            backward.append(self.hs_backward)
            
        # LSTM
        for fwd, bwd in zip(forward, backward):
            input_tensor = torch.cat((fwd, bwd), 1)
            self.hs_lstm, self.cs_lstm = self.lstm_cell(input_tensor, (self.hs_lstm, self.cs_lstm))
            
        # 最后一个隐藏状态通过线性层
        out = self.linear(self.hs_lstm)
        
        return out

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model
