from time import time
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,accuracy_score, make_scorer, precision_score

from vectorization import VectorizationMethodManager  
from classification import ClassificationMethodManager

class Trainer:
    def __init__(self, X, y):
        self.vectorizationManager = VectorizationMethodManager() 
        self.classificationManager = ClassificationMethodManager()
        self.X = X
        self.y = y

    def train(self,  vectMethodArgs, classificationMethodArgs):
        start = time()
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        vectorizer = self.vectorizationManager.get_method(vectMethodArgs)
        classifier = self.classificationManager.get_method(classificationMethodArgs)
        pipe = Pipeline([(vectMethodArgs['Vectorizer'].name, vectorizer), (classificationMethodArgs['Classifier'].name, classifier)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        print(classification_report(y_test, y_pred))
        print('Accuracy: ', accuracy_score(y_test, y_pred))
        print('Elapsed Time: ', time()-start)
        return pipe

    def grid_search(self, vectMethodArgs, classificationMethodArgs, GridArgs):
        start = time()
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        vectorizer = self.vectorizationManager.get_method(vectMethodArgs)
        classifier = self.classificationManager.get_method(classificationMethodArgs)
        vectorizerName = vectMethodArgs['Vectorizer'].name
        classifierName = classificationMethodArgs['Classifier'].name
        pipeline = Pipeline([(vectorizerName, vectorizer), (classifierName, classifier)])
        print('Grid Search is running with '+vectorizerName+' and '+classifierName) 
        grid = GridSearchCV(pipeline, GridArgs, cv=5, scoring='f1_weighted', verbose=4, n_jobs=-1,  ) 
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        print(classification_report(y_test, y_pred))
        print('Accuracy: ', accuracy_score(y_test, y_pred))
        print('Elapsed Time: ', time()-start)
        return grid.best_estimator_
    
    def save_model(self, model, filename): 
        joblib.dump(model, filename)
        print('Model is saved as '+filename)

    def load_model(self, filename):
        model = joblib.load(filename)
        print('Model is loaded from '+filename)
        return model

    