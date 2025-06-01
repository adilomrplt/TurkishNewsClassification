from enum import Enum
from sklearn.feature_extraction.text import TfidfVectorizer 


class Vectorizer(Enum):
    TFIDF = 0


class VectorizationMethodManager:
    def __init__(self):
        pass 
        

    def get_method(self, methodArgs):
        methodName = methodArgs['Vectorizer'].name
        if methodName == 'TFIDF':
            return   TfidfVectorizer(stop_words=methodArgs['stop_words'])
                           
        else:
            raise ValueError('Method not recognized')