import numpy as np

class Encode:
    '''encode str to int'''
    def encoding(self, vocab, tokenized) -> list:

        word_to_index = {}
        word_to_index['<PAD>'] = 0
        word_to_index['<UNK>'] = 1

        for index, word in enumerate(vocab) :
            word_to_index[word] = index + 2

        encoded = Encode.texts_to_sequences(tokenized, word_to_index)

        return encoded
    
    @staticmethod
    def texts_to_sequences(tokenized_X_data, word_to_index):
        encoded_X_data = []
        for sent in tokenized_X_data:
            index_sequences = []
            for word in sent:
                try:
                    index_sequences.append(word_to_index[word])
                except KeyError: #OOV
                    index_sequences.append(word_to_index['<UNK>'])
                encoded_X_data.append(index_sequences)
        return encoded_X_data
    
    '''make int padding'''
    def padding(self, sentences, max_len) -> list:
        features = np.zeros((len(sentences), max_len), dtype=int)
        for index, sentence in enumerate(sentences):
            if len(sentence) != 0:
                features[index, :len(sentence)] = np.array(sentence)[:max_len]
        return features