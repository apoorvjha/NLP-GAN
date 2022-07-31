from os.path import exists
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from numpy import array

class Tokenize:
    def __init__(self,num_words=None):
        self.num_words=num_words
        if self.num_words!=None:
            self.tokenizer=Tokenizer(num_words=self.num_words)
        else:
            self.tokenizer=Tokenizer()
    def fit(self,words):
        self.tokenizer.fit_on_texts(words)
    def getSequences(self,inputs):
        return self.tokenizer.texts_to_sequences(inputs)
    def padSequences(self,seq,maxlen=250,padding='pre'):
        return pad_sequences(seq,maxlen=maxlen,padding=padding)
    def save(self,fname='tokenizer.json'):
        #print(self.tokenizer.to_json())
        with open(fname,'w') as config_file:
            json.dump(self.tokenizer.to_json(),config_file)
    def load_from_config(self,fname='tokenizer.json'):
        with open(fname,'r') as config_file:
            config=json.load(config_file)
        self.tokenizer = tokenizer_from_json(config)
    def getVocabSize(self):
        return len(self.tokenizer.word_index.keys())
    def getVocab(self):
        return list(self.tokenizer.word_index.keys())

class Preprocessing:
    def __init__(self):
        self.path = './POC_data/eng_fra.txt'
        self.tokenizer_input_config = './tokenizer_input.json'
        self.tokenizer_output_config = './tokenizer_output.json'
        self.read_data()
        self.preprocess_data()
        if exists(self.tokenizer_input_config) and exists(self.tokenizer_output_config):
            self.tokenizer_input = Tokenize()
            self.tokenizer_input.load_from_config(self.tokenizer_input_config)
            self.tokenizer_output = Tokenize()
            self.tokenizer_output.load_from_config(self.tokenizer_output_config) 
        else:
            self.tokenizer_input = Tokenize()
            self.tokenizer_output = Tokenize()
            self.tokenize()
            
    def read_data(self):
        self.data = []
        try:
            with open(self.path, 'r', encoding='utf-8') as fd:
                raw_data = fd.read()
        except Exception as e:
            print(e)
        else:
            for i in raw_data.split('\n'):
                self.data.append(i.split('\t'))
        finally:
            return self.data

    def remove_stop_symbols(self, words):
        stop_symbols = ['.','!','@','&','?',',','\'']
        for i in range(len(words)):
            for symbol in stop_symbols:
                words[i] = words[i].replace(symbol,'')
        return words

    def tokenize(self):
        input_words = []
        output_words = []
        for i in range(len(self.data)):
            input_sentence = self.data[i][0]
            output_sentence = self.data[i][1]
            for word in input_sentence.split(' '):
                if word not in input_words and len(word) != 0:
                    input_words.append(word)
            for word in output_sentence.split(' '):
                if word not in output_words and len(word) != 0:
                    output_words.append(word)
        self.tokenizer_input.fit(input_words)
        self.tokenizer_output.fit(output_words)
        self.tokenizer_input.save(fname=self.tokenizer_input_config)
        self.tokenizer_output.save(fname=self.tokenizer_output_config)
    
    def generateTokens(self, sentence, mode=0, is_pad=True):
        data = ' '.join(self.remove_stop_symbols(sentence.split(' ')))
        if mode == 0:
            sequence = self.tokenizer_input.getSequences([data])
            if is_pad:
                sequence = self.tokenizer_input.padSequences(sequence)
        else:
            sequence = self.tokenizer_output.getSequences([data])
            if is_pad:
                sequence = self.tokenizer_output.padSequences(sequence)
        return array(sequence).reshape(-1,1)

    def addSentenceTokens(self, data):
        for i in range(len(data)):
            data[i] = '<SOS> ' + data[i] + ' <EOS>'
        return data

    def preprocess_data(self):
        '''
        TODO:
            1. remove the occurances of native names from the sentences.
        '''
        self.data = list(map(self.remove_stop_symbols, self.data))
        self.data.remove([''])
        self.data = list(map(self.addSentenceTokens, self.data))
        return self.data

    def get_X_Y(self):
        X = []
        Y = []
        for i in range(len(self.data)):
            input_sentence = self.data[i][0]
            output_sentence = self.data[i][1]
            X.append(self.generateTokens(input_sentence))
            Y.append(self.generateTokens(output_sentence,mode=1))
        return array(X), array(Y)

    def getVocabSize(self):
        return self.tokenizer_input.getVocabSize(), self.tokenizer_output.getVocabSize()

    def get_SOS_EOS_token(self):
        return self.tokenizer_output.tokenizer.word_index['sos'], self.tokenizer_output.tokenizer.word_index['eos']
    
