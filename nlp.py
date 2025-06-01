import nltk
from nltk.corpus import stopwords

class NLPManager:
    def __init__(self):
        nltk.download('stopwords')
        self.stop_words = stopwords.words('turkish')

    def remove_stop_words(self, text):
        return ' '.join([word for word in text.split() if word not in self.stop_words])

