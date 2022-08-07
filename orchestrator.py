import preprocessing
import GAN
from random import randint
import torch
import time
from numpy import array

if __name__ == '__main__':
    preprocessor = preprocessing.Preprocessing()
    X, Y = preprocessor.get_X_Y()
    # X = torch.from_numpy(X).type(torch.LongTensor)
    # Y = torch.from_numpy(Y).type(torch.LongTensor)
    input_vocab_size, output_vocab_size = preprocessor.getVocabSize()
    hidden_dim = 128
    n_classes = 1
    max_sequence_length = max([i.shape[0] for i in X]) - 1
    SOS, EOS = preprocessor.get_SOS_EOS_token()
    print("Shapes : ", X.shape, Y.shape)
    print("max_length : ",max_sequence_length)
    print("Vocb_sizes : ", input_vocab_size, output_vocab_size)
    print("SOS and EOS : ", SOS, EOS)
    print("Hidden Dim : ", hidden_dim)
    gan = GAN.NLP_GAN(input_vocab_size, hidden_dim, output_vocab_size, max_sequence_length, n_classes, SOS, EOS)
    start = time.time()
    gan.train(X[:100], Y[:100], 2)
    print(time.time() - start)
    # idx = randint(0,X.shape[0]-11)
    # print(X[idx : idx + 10])
    # print(Y[idx : idx + 10])
    test = "I am a cool man!"
    X = array(preprocessor.generateTokens(test))
    Y = gan.generate(X)
    if SOS in Y:
        Y.remove(SOS)
    if EOS in Y:
        Y.remove(EOS)
    print(preprocessor.decodeSequence(Y))

    
    
