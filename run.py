from time import time
import numpy as np 

from data import DataManager 
from vectorization import Vectorizer
from classification import Classifier
from nlp import NLPManager
from train import Trainer


dm = DataManager('data/news.json')                                  
dm.clean_data()


trainer = Trainer(X=dm.get_X_Vectors(), y=dm.get_Y())
 

# Vectorizer Parametreleri 
nlp = NLPManager()
tfidfParam = {'Vectorizer':Vectorizer.TFIDF, 'stop_words': nlp.stop_words}


# Classifier Parametreleri
dtreeParam = {'Classifier':Classifier.DTREE}
knnParam = {'Classifier':Classifier.KNN}
sgdParam = {'Classifier':Classifier.SGD}
nbParam = {'Classifier':Classifier.NB}
svmParam = {'Classifier':Classifier.SVM}
randforestParam = {'Classifier':Classifier.RANDFOREST}
logregParam = {'Classifier':Classifier.LOGREG}


gridParamDecisionTree = { 
    'TFIDF__max_features': [5000, 10000, 15000],
    'TFIDF__ngram_range': [(1, 1), (1, 2)], 
    'DTREE__max_depth': [10, 20, 30],  # Karar ağacının derinliği
    'DTREE__min_samples_split': [2, 5, 10],  # Düğüm bölünmeleri için minimum örnek sayısı
    'DTREE__min_samples_leaf': [1, 2, 4],  # Yapraklar için minimum örnek sayısı
    'DTREE__criterion': ['gini', 'entropy'],  # Kullanılacak kriterler (Gini veya Entropi)
}

gridParamKNN = { 
    'TFIDF__max_features': [5000, 10000, 15000],
    'TFIDF__ngram_range': [(1, 1), (1, 2)],
    'KNN__n_neighbors': [3, 5, 7, 10],  # k-NN için komşu sayısı parametresi
    'KNN__metric': ['euclidean', 'manhattan'],  # Mesafe ölçütü
}    

gridParamSGD = { 
    'TFIDF__max_features': [5000, 10000, 15000],
    'TFIDF__ngram_range': [(1, 1), (1, 2)],    
    'SGD__alpha': [0.0001, 0.001, 0.01],  # SGD için alpha parametresi
    'SGD__loss': ['log_loss','hinge', 'perceptron'],  # Kayıp fonksiyonu
    'SGD__penalty': ['l2'],  # Düzenleme terimi
}
    
gridParamNaiveBayes = { 
    'TFIDF__max_features': [5000, 10000, 15000],
    'TFIDF__ngram_range': [(1, 1), (1, 2)],
    'NB__alpha': [0.5, 1, 2],  # Naive Bayes için alpha parametresi
    'NB__fit_prior': [True, False],  # Önceki olasılıkların öğrenilip öğrenilmeyeceği
}

gridParamSVM = { 
    'TFIDF__max_features': [5000, 10000, 15000],
    'TFIDF__ngram_range': [(1, 1), (1, 2)],
    'SVM__C': [0.1, 1, 10],  # SVM'in 'C' hiperparametresi
    'SVM__kernel': ['linear', 'rbf'],  # Farklı kernel fonksiyonları ile test etme
}

gridParamRandomForest = { 
    'TFIDF__max_features': [5000, 10000, 15000],
    'TFIDF__ngram_range': [(1, 1), (1, 2)],
    'RANDFOREST__n_estimators': [100, 200, 300],  # Karar ağacı sayısı
    'RANDFOREST__max_depth': [10, 20, 30],  # Karar ağacının derinliği
    'RANDFOREST__min_samples_split': [2, 5, 10],  # Düğüm bölünmeleri için minimum örnek sayısı
    'RANDFOREST__min_samples_leaf': [1, 2, 4],  # Yapraklar için minimum örnek sayısı
    'RANDFOREST__criterion': ['gini', 'entropy'],  # Kullanılacak kriterler (Gini veya Entropi)
}

gridParamLogisticRegression = { 
    'TFIDF__max_features': [5000, 10000, 15000],
    'TFIDF__ngram_range': [(1, 1), (1, 2)], 
    'LOGREG__C': [0.1, 1, 10],  # C hiperparametresi
    'LOGREG__penalty': ['l2'],  # Düzenleme terimi
    'LOGREG__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],  # Optimizasyon algoritması
    'LOGREG__max_iter': [100, 300, 500],  # Maksimum iterasyon sayısı
}



start = time()
#best_dtree_model =  trainer.grid_search(tfidfParam, dtreeParam, gridParamDecisionTree)
#trainer.save_model(best_dtree_model, 'models/best_dtree_model.pkl')
best_knn_model = trainer.grid_search(tfidfParam, knnParam, gridParamKNN)
trainer.save_model(best_knn_model, 'models/best_knn_model.pkl')
#best_sgd_model = trainer.grid_search(tfidfParam, sgdParam, gridParamSGD)
#trainer.save_model(best_sgd_model, 'models/best_sgd_model.pkl')
#best_nb_model = trainer.grid_search(tfidfParam, nbParam, gridParamNaiveBayes)
#trainer.save_model(best_nb_model, 'models/best_nb_model.pkl')
#best_svm_model = trainer.grid_search(tfidfParam, svmParam, gridParamSVM)
#trainer.save_model(best_svm_model, 'models/best_svm_model.pkl')
#best_randomforest_model = trainer.grid_search(tfidfParam, randforestParam, gridParamRandomForest)
#trainer.save_model(best_randomforest_model, 'models/best_randomforest_model.pkl')
#best_logisticregression_model = trainer.grid_search(tfidfParam, logregParam, gridParamLogisticRegression)
#trainer.save_model(best_logisticregression_model, 'models/best_logisticregressin_model.pkl')

print('Elapsed Time: ', time()-start)
