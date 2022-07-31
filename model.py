import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.optim import SGD
import random

# if torch.cuda.is_available():
#     device = 'cuda'
# else:
#     device = 'cpu'

device = 'cpu'

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
    def forward(self, X, hidden):
        X = self.embedding(X)
        X = X.view(1,1,-1)
        X, hidden = self.gru(X, hidden)
        return X, hidden
    def initHidden(self):
        return torch.zeros(1,1,self.hidden_dim, device = device)

class Decoder(nn.Module):
    '''
    TODO : 
        1. Code out the attention mechanism.
    '''
    def __init__(self, output_dim,hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.hidden_dim = hidden_dim 
    def forward(self, X, hidden):
        X = self.embedding(X)
        X = F.relu(X.view(1,1,-1))
        X, hidden = self.gru(X, hidden)
        X = self.softmax(self.linear(X[0]))
        return X, hidden
    def initHidden(self):
        return torch.zeros(1,1,self.hidden_dim, device = device)

class Generator(nn.Module):
    '''
        TODO :
            1. Fix the vlock of code so that we get gradients to pass through. This requires 
            passing OHV representation of output tokens. 
    '''
    def __init__(self, input_dim, hidden_dim, output_dim, max_sequence_length, SOS, EOS):
        super(Generator, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim).to(device)
        self.decoder = Decoder(output_dim, hidden_dim).to(device)
        self.max_sequence_length = max_sequence_length
        self.SOS = SOS
        self.EOS = EOS
        learning_rate = 1e-3
        self.criterion = nn.CrossEntropyLoss()
        self.encoder_optimizer = SGD(self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = SGD(self.decoder.parameters(), lr=learning_rate)
        self.teacher_forcing_ratio = 1.0 # the orignal tokens are passed.
    def forward(self, X):
        e_hidden = self.encoder.initHidden()
        input_length = X.size(0)
        for i in range(input_length):
            e_output, e_hidden = self.encoder(X[i], e_hidden)
        #d_hidden = self.decoder.initHidden()
        d_hidden = e_hidden
        output_sequences = []
        X = torch.tensor([[self.SOS]],device=device)
        output_sequences.append(self.SOS)
        for i in range(self.max_sequence_length):
            X, d_hidden = self.decoder(X, d_hidden)
            topv, topi = X.topk(1)
            X = topi.squeeze().detach()
            output_sequences.append(X.item())
            if X.item() == self.EOS:
                break
        return output_sequences

    def train(self, Xs, Ys):
        for X, Y in zip(Xs,Ys):
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            e_hidden = self.encoder.initHidden()
            input_length = X.size(0)
            target_length = Y.size(0)
            for i in range(input_length):
                e_output, e_hidden = self.encoder(X[i], e_hidden)
            #d_hidden = self.decoder.initHidden()
            d_hidden = e_hidden
            output_sequence = []
            loss = 0
            Xi = torch.tensor([[self.SOS]],device=device)
            use_teacher_forcing = True if random.random() > self.teacher_forcing_ratio else False
            if use_teacher_forcing:
                for i in range(target_length):
                    Xi, d_hidden = self.decoder(Xi, d_hidden)
                    # ---------------------- To be fixed ------------- #
                    # topv, topi = Xi.topk(1)
                    # Xi = topi.squeeze().detach()
                    #--------------------------------------------------#
                    loss += self.criterion(Xi, Y[i])
                    Xi = Y[i]                
            else:
                for i in range(target_length):
                    Xi, d_hidden = self.decoder(Xi, d_hidden)
                    # ---------------------- To be fixed ------------- #
                    # topv, topi = Xi.topk(1)
                    # Xi = topi.squeeze().detach()
                    #--------------------------------------------------#
                    loss += self.criterion(Xi.view(-1,1).type(torch.FloatTensor), Y[i].view(-1))
                    if Xi.item() == self.EOS:
                        break
            loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            print(loss.item() / target_length)

class Dense(nn.Module):
    def __init__(self, input_dim, output_dim,hidden_layers=[8]):
        super(Dense, self).__init__()
        layers = []
        if len(hidden_layers) == 0:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_layers[0]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            for layer in range(1,len(hidden_layers)):
                layers.append(nn.Linear(hidden_layers[layer - 1], hidden_layers[layer]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=0.2))
            layers.append(nn.Linear(hidden_layers[-1], output_dim))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
    def forward(self, X):
        X = self.model(X)
        return X
        
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''
        TODO :
            1. Create a seq2vec classification model. : Done
            2. Write the training loop for the same.  : Done
        ''' 
        super(Discriminator, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim).to(device)
        learning_rate = 1e-3
        self.criterion = nn.BCELoss()
        self.encoder_optimizer = SGD(self.encoder.parameters(), lr=learning_rate)
        self.dense = Dense(hidden_dim, output_dim)
        self.dense_optimizer = SGD(self.dense.parameters(), lr=learning_rate)
    def forward(self, X):
        e_hidden = self.encoder.initHidden()
        input_length = X.size(0)
        for i in range(input_length):
            e_output, e_hidden = self.encoder(X[i], e_hidden)
        return self.dense(e_output)
    def train(self, Xs, Ys):
        for X, Y in zip(Xs, Ys):
            self.encoder_optimizer.zero_grad()
            self.dense_optimizer.zero_grad()
            e_hidden = self.encoder.initHidden()
            loss = 0
            input_length = X.size(0)
            for i in range(input_length):
                e_output, e_hidden = self.encoder(X[i], e_hidden)
            prediction = self.dense(e_output)[0]
            Y = Y.to(device)
            print(prediction, Y)
            loss += self.criterion(prediction, Y)
            loss.backward()
            self.encoder_optimizer.step()
            self.dense_optimizer.step()
            print(loss.item())


def test():
    N=200
    # G = Generator(250, 64, 500, 10).to(device)
    # print(G)
    # X = torch.zeros(N,20,1, dtype=torch.long, device=device)
    # Y = torch.zeros(N,30,1, dtype=torch.long, device=device)
    # G.train(X, Y)
    # d = Dense(100,2)
    # print(d)
    # D = Discriminator(250, 64, 1).to(device)
    # print(D)
    # Y = torch.zeros(N, 1, dtype=torch.long, device=device)
    # D.train(X, Y)

if __name__ == '__main__':
    # test()
    pass



