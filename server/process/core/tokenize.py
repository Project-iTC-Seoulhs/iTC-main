from konlpy.tag import Okt

class Tokenize:
    
    '''tokenize english sentence'''
    def tokenize(self, text:str) -> list:
        tokenized = Okt.morphs(text)

        return tokenized


    '''remove stopwords'''
    def stopword(self, text_tokenized :list) -> list:
        result = []

        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english')) 

        for a in text_tokenized:
            if a not in stop_words:
                result.append(a)
                
        return result
    
    
