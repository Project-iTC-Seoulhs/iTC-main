from collections import Counter

class Vocab:
    '''make text vocab'''
    def vocab(self, tokenized_X_train, threshold) -> dict:
        word_list = []

        total_cnt = len(word_counts)

        for sent in tokenized_X_train:
            for word in sent:
                word_list.append(word)

        word_counts = Counter(word_list)

        vocab = sorted(word_counts, key=word_counts.get, reverse=True)

        threshold = 3

        rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
        total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
        rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

        for key, value in word_counts.items():
            total_freq = total_freq + value

            if(value < threshold):
                rare_cnt = rare_cnt + 1
                rare_freq = rare_freq + value

            vocab_size = total_cnt - rare_cnt
            vocab = vocab[:vocab_size]

        return vocab