import model
from numpy import array

class NLP_GAN:
    def __init__(self, input_vocab_size, hidden_dim, output_vocab_size, max_sequence_length, n_classes, SOS, EOS):
        self.Generator = model.Generator(input_vocab_size, hidden_dim, output_vocab_size, max_sequence_length, SOS, EOS).to(model.device)
        self.Discriminator = model.Discriminator(output_vocab_size, hidden_dim, n_classes).to(model.device)
        self.k = 1
    def train(self, X, Y, epochs):
        X = model.torch.tensor(X).type(model.torch.LongTensor).to(model.device)
        Y = model.torch.tensor(Y).type(model.torch.LongTensor).to(model.device)
        for epoch in range(epochs):
            print(f"EPOCH[{epoch}]")
            for Xi, Yi in zip(X,Y):
                labels = []
                inputs = []
                fake_data = array(self.Generator(Xi)).reshape(-1,1)
                inputs.append(fake_data)
                labels.append(1)
                orignal_data = Yi.cpu().numpy()
                inputs.append(orignal_data)
                labels.append(0)
                inputs = array(inputs)
                labels = array(labels)
                inputs = model.torch.tensor(inputs, device=model.device)
                labels = model.torch.tensor(labels, device=model.device).view(-1,1,1).type(model.torch.FloatTensor)
                for iteration in range(self.k):
                    self.Discriminator.train(inputs, labels)
                self.Generator.train(Xi, Yi)
    def generate(self, X):
        return self.Generator(X)
    

