import model

class NLP_GAN:
    def __init__(self, input_vocab_size, hidden_dim, output_vocab_size, max_sequence_length, n_classes):
        self.Generator = model.Generator(input_vocab_size, hidden_dim, output_vocab_size, max_sequence_length)
        self.Discriminator = model.Discriminator(output_vocab_size, hidden_dim, n_classes)
        self.k = 1
    def train(self, X, Y, epochs):
        for epoch in epochs:
            labels = []
            inputs = []
            for Xi, Yi in zip(X,Y):
                fake_data = self.Generator(Xi)
                inputs.append(fake_data)
                labels.append(1)
                orignal_data = Yi
                inputs.append(orignal_data)
                labels.append(0)
            inputs = torch.tensor(inputs)
            labels = torch.tensor(labels)
            for iteration in range(self.k):
                self.Discriminator.train(inputs, labels)
            self.Generator.train(Xi, Yi)
    def generate(self, X):
        return self.Generator(X)
    

