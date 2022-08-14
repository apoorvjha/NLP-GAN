import preprocessing
import GAN
from random import randint
import torch
import time
from numpy import array
from sklearn.model_selection import train_test_split
from torchtext.data.metrics import bleu_score
import numpy as np

if __name__ == '__main__':
    preprocessor = preprocessing.Preprocessing()
    X, Y = preprocessor.get_X_Y()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.5, shuffle=True, random_state=42)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
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
    gan = GAN.NLP_GAN(input_vocab_size + 1, hidden_dim, output_vocab_size + 1, max_sequence_length, n_classes, SOS, EOS)
    start = time.time()
    gan.train(X_train, Y_train, 5)
    print("Training time : ",time.time() - start)
    predictions=[]
    labels = []
    Y_test = Y_test.reshape(Y_test.shape[0],-1)
    for test_input, test_label in zip(X_test, Y_test):
        prediction = gan.generate(test_input)
        if SOS in prediction:
            prediction.remove(SOS)
        if EOS in prediction:
            prediction.remove(EOS)
        prediction = preprocessor.decodeSequence(prediction)[0].split(' ')
        test_label = np.delete(test_label,np.where(test_label == 0))
        label = preprocessor.decodeSequence(test_label)[0].split(' ')
        label.remove('eos')
        label.remove('sos')
        predictions.append(prediction)
        labels.append(label)
    
    print("blue score : ",bleu_score(predictions, labels))

    

    
    
