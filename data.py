import re

import pandas as pd

from nlp import NLPManager

class DataManager:
    def __init__(self, data_path):
        self.nlp_manager = NLPManager()  
        loaded = pd.read_json(data_path)
        self.data =loaded.dropna(subset=['Durum'])
        print(self.data.info())
    
    def clean_data(self):  
        self.data['Puan'] = self.data['Durum'].astype('int')
        self.data['Body'] = self.data['Body'].astype(str).str.strip()  # Metin sütununu temizle ve string türüne çevir
        self.data['Body'] = self.data['Body'].apply(lambda x: re.sub(r'\W', ' ', x.lower()))  # Küçük harfe dönüştürme ve özel karakterleri temizleme
        self.data.info()

    def get_X_Vectors(self):
        return self.data['Body'] 
    
    def get_Y(self):
        return self.data['Durum']
    
    def get_Y_Numeric(self):
        return self.data['Puan']
    
