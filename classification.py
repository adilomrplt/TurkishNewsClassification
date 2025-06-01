from enum import Enum
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class Classifier(Enum):
    DTREE = 0
    KNN = 1
    SGD = 2
    NB = 3
    SVM = 4
    RANDFOREST = 5
    LOGREG = 6


class ClassificationMethodManager:
    def __init__(self):
        pass 
        

    def get_method(self, methodArgs):
        methodName = methodArgs['Classifier'].name
        if methodName == 'DTREE':
            return DecisionTreeClassifier(random_state=42)
        elif methodName == 'KNN':
            return KNeighborsClassifier()
        elif methodName == 'SGD':
            return SGDClassifier(random_state=42)
        elif methodName == 'NB':    
            return MultinomialNB()
        elif methodName == 'SVM':
            return SVC(random_state=42)
        elif methodName == 'RANDFOREST':
            return RandomForestClassifier(random_state=42)
        elif methodName == 'LOGREG':
            return LogisticRegression(random_state=42)
        else:
            raise ValueError('Method not recognized')