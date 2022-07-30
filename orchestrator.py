import preprocessing
import GAN
from random import randint
import torch

if __name__ == '__main__':
    preprocessor = preprocessing.Preprocessing()
    X, Y = preprocessor.get_X_Y()
    # idx = randint(0,X.shape[0]-11)
    # print(X[idx : idx + 10])
    # print(Y[idx : idx + 10])
    X = torch.tensor(X)
    print(X)
    
