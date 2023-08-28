import numpy as np
from sms_spam_detection.entity.config_entity import TrainingConfig
from sms_spam_detection.utils.common import save_bin
from sklearn.metrics import r2_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score

class ModelTrainer:
    def __init__(self, config:TrainingConfig):
        self.trainer_config=config
        
    def evaluate_models(self, X_train, y_train,X_test,y_test,models,param):
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report


    def initiate_model_trainer(self,X_train,y_train,X_test,y_test):
        models = {
            "Naive Bayes": MultinomialNB(),
            "Logistic Regression": LogisticRegression(),
            "Support Vector Machines": SVC()
        }
        params={
            "Naive Bayes": {
                "alpha": [0.1,0.3, 0.6, 1.0]
            },
            "Logistic Regression": {
                "C":np.logspace(-3,3,7),
                "penalty":["l1","l2"]
            },
            "Support Vector Machines": {
                'C': [0.1, 1, 10, 100, 1000],
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel': ['rbf']
            }
        }

        model_report:dict=self.evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                            models=models,param=params)
        
        best_model_score = max(sorted(model_report.values()))

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]

        save_bin(
            path=self.trainer_config.trained_model_path,
            data=best_model
        )

        predicted=best_model.predict(X_test)

        r2_square = r2_score(y_test, predicted)
        accuracy=accuracy_score(y_test, predicted)
        precision=precision_score(y_test, predicted)
        recall=recall_score(y_test, predicted)
        result={"r2_square": r2_square,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall}
        return result