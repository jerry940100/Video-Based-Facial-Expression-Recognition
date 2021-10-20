from numpy.core.arrayprint import set_string_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# reference https://clay-atlas.com/blog/2020/06/09/pytorch-cn-note-nn-embedding-layer-convert-vector/
#           https://clay-atlas.com/blog/2020/05/12/pytorch-lstm-%E7%9A%84%E5%8E%9F%E7%90%86%E8%88%87%E8%BC%B8%E5%85%A5%E8%BC%B8%E5%87%BA%E6%A0%BC%E5%BC%8F%E7%B4%80%E9%8C%84/


class MobileNet_Extractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.mobilenet = models.mobilenet_v3_large(pretrained)
        del self.mobilenet.classifier
        features = list(self.mobilenet.features)
        features.append(self.mobilenet.avgpool)
        self.layers = nn.ModuleList(features).eval()

    def forward(self, x):
        # x shape[batch,depth,3,224,224]
        output = torch.zeros((x.shape[0], x.shape[1], 960)).cuda()
        for d in range(x.shape[1]):
            x_temp = x[:, d]
            for ii, model in enumerate(self.layers):
                x_temp = model(x_temp)
            x_temp = x_temp.view(-1, 960)
            output[:, d] = x_temp
        # x shape[batch,depth,seq_length]
        return output


class LSTM_Attention(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, n_layers, num_class):

        super(LSTM_Attention, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 单词数，嵌入向量维度
        self.encoder = MobileNet_Extractor()
        self.rnn = nn.LSTM(embedding_dim, hidden_dim,
                           num_layers=n_layers, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_class)
        self.dropout = nn.Dropout(0.5)

        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(
            torch.Tensor(hidden_dim, hidden_dim)).cuda()
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim, 1)).cuda()

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def attention_net(self, x):  # x:[batch, seq_len, hidden_dim]

        # [batch, seq_len, hidden_dim*2]
        u = torch.tanh(torch.matmul(x, self.w_omega))
        att = torch.matmul(u, self.u_omega)  # [batch, seq_len, 1]
        att_score = F.softmax(att, dim=1)

        scored_x = x * att_score  # [batch, seq_len, hidden_dim]

        context = torch.sum(scored_x, dim=1)  # [batch, hidden_dim]
        return context

    def forward(self, x):
        embedding = self.encoder(x)
        print(embedding.shape)
        # output: [seq_len, batch, hidden_dim]     hidden/cell: [n_layers*2, batch, hidden_dim]
        output, (final_hidden_state, final_cell_state) = self.rnn(embedding)
        # output = output.permute(1, 0, 2)  # [batch, seq_len, hidden_dim*2]

        attn_output = self.attention_net(output)
        logit = self.fc(attn_output)
        return logit


"""
class BiLSTM_Attention(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers):

        super(BiLSTM_Attention, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 单词数，嵌入向量维度
        self.rnn = nn.LSTM(embedding_dim, hidden_dim,
                           num_layers=n_layers, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.5)

        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(
            torch.Tensor(hidden_dim * 2, hidden_dim * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def attention_net(self, x):  # x:[batch, seq_len, hidden_dim*2]

        # [batch, seq_len, hidden_dim*2]
        u = torch.tanh(torch.matmul(x, self.w_omega))
        att = torch.matmul(u, self.u_omega)  # [batch, seq_len, 1]
        att_score = F.softmax(att, dim=1)

        scored_x = x * att_score  # [batch, seq_len, hidden_dim*2]

        context = torch.sum(scored_x, dim=1)  # [batch, hidden_dim*2]
        return context

    def forward(self, x):
        # [seq_len, batch, embedding_dim]
        embedding = self.dropout(self.embedding(x))

        # output: [seq_len, batch, hidden_dim*2]     hidden/cell: [n_layers*2, batch, hidden_dim]
        output, (final_hidden_state, final_cell_state) = self.rnn(embedding)
        output = output.permute(1, 0, 2)  # [batch, seq_len, hidden_dim*2]

        attn_output = self.attention_net(output)
        logit = self.fc(attn_output)
        return logit
"""
if __name__ == "__main__":
    model = LSTM_Attention(embedding_dim=960, hidden_dim=960, n_layers=1)
    input = torch.randn((2, 8, 3, 224, 224))
    output = model(input)
    print(output)
