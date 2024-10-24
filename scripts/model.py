import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn

mlflow.sklearn.autolog()
def train_and_evaluate(data,model):
    """
    This function is a big function that does several tasks with one run. This function does the following
      - Divides the feature and target variables
      - Performs a train test split function on the variables
      - Trains the muliple models according to a machine learning model mentioned as a parameter
      - Evaluates the performace of the models based on 4 metrics (accuracy, precision, recall and f1)
      - Logs every metrics, parameters for the models and the models themselves to the mlflow 
    Parameters:
      1. data- for our case, it's either the fraud dataset or the credit card dataset
      2. model- model chosen to train the data with. 
    Returns: 
      - The evaluation metrics and the models logged into the mlflow
    """
    if data=='fraud_data':
        data= pd.read_csv('data/fraud_transformed.csv')
        mlflow.set_experiment("Fraud Detection with Fraud Dataset")
        if model == 'LogisticRegression':
            with mlflow.start_run(run_name='Logistic Regression with Fraud'):
                lr_model=LogisticRegression(C=0.01,max_iter=10,solver='saga')
                X=data.drop(['Unnamed: 0','class'],axis=1)
                y=data['class']
                X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
                lr_model.fit(X_train,y_train)
                y_pred=lr_model.predict(X_test)
                accuracy=round(accuracy_score(y_pred,y_test),6)
                precision=round(precision_score(y_pred,y_test),6)
                recall=round(recall_score(y_pred,y_test),6)
                f1=round(f1_score(y_pred,y_test),6)
                print(f"The following metrics are found when the fraud dataset is trained with {model} model")
                print('------------------------')
                print(f"The accuracy is {accuracy}")
                print('------------------------')
                print(f"The precision is {precision}")
                print('------------------------')
                print(f"The recall is {recall}")
                print('------------------------')
                print(f"The f1 score is {f1}")
                print('------------------------')
                #logging the metrics
                mlflow.log_metric('accuracy',accuracy)
                mlflow.log_metric('precision',precision)
                mlflow.log_metric('recall',recall)
                mlflow.log_metric('f1',f1)
                #logging the model itself
                mlflow.sklearn.log_model(lr_model, "lr_model_with_fraud_data")
                #logging the parameters used
                mlflow.log_param('C',0.01)
                mlflow.log_param('max_iter',10)
                mlflow.log_param('solver','saga')
        elif model == 'DecisionTree':
            with mlflow.start_run(run_name='Decision Tree with Fraud'):
                dt_model=DecisionTreeClassifier(max_depth=10,min_samples_leaf=10,max_leaf_nodes=100)
                X=data.drop(['Unnamed: 0','class'],axis=1)
                y=data['class']
                X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
                dt_model.fit(X_train,y_train)
                y_pred=dt_model.predict(X_test)
                accuracy=round(accuracy_score(y_pred,y_test),6)
                precision=round(precision_score(y_pred,y_test),6)
                recall=round(recall_score(y_pred,y_test),6)
                f1=round(f1_score(y_pred,y_test),6)
                print(f"The following metrics are found when the fraud dataset is trained with {model} model")
                print('------------------------')
                print(f"The accuracy is {accuracy}")
                print('------------------------')
                print(f"The precision is {precision}")
                print('------------------------')
                print(f"The recall is {recall}")
                print('------------------------')
                print(f"The f1 score is {f1}")
                print('------------------------')
                #logging the metrics
                mlflow.log_metric('accuracy',accuracy)
                mlflow.log_metric('precision',precision)
                mlflow.log_metric('recall',recall)
                mlflow.log_metric('f1',f1)
                #logging the model itself
                mlflow.sklearn.log_model(dt_model, "dt_model_with_fraud_data")
                #logging the parameters used
                mlflow.log_param("max_depth", 10)
                mlflow.log_param("min_samples_leaf", 10)
                mlflow.log_param("max_leaf_nodes", 100)
        elif model == 'RandomForest':
            with mlflow.start_run(run_name='Random Forest with Fraud'):
                rf_model=RandomForestClassifier(max_depth=10,n_estimators=100)
                X=data.drop(['Unnamed: 0','class'],axis=1)
                y=data['class']
                X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
                rf_model.fit(X_train,y_train)
                y_pred=rf_model.predict(X_test)
                accuracy=round(accuracy_score(y_pred,y_test),6)
                precision=round(precision_score(y_pred,y_test),6)
                recall=round(recall_score(y_pred,y_test),6)
                f1=round(f1_score(y_pred,y_test),6)
                print(f"The following metrics are found when the fraud dataset is trained with {model} model")
                print('------------------------')
                print(f"The accuracy is {accuracy}")
                print('------------------------')
                print(f"The precision is {precision}")
                print('------------------------')
                print(f"The recall is {recall}")
                print('------------------------')
                print(f"The f1 score is {f1}")
                print('------------------------')
                #logging the metrics
                mlflow.log_metric('accuracy',accuracy)
                mlflow.log_metric('precision',precision)
                mlflow.log_metric('recall',recall)
                mlflow.log_metric('f1',f1)
                #logging the model itself
                mlflow.sklearn.log_model(rf_model,'rf_model_with_fraud_data')
                #logging the parameters used
                mlflow.log_param('max_depth',10)
                mlflow.log_param('n_estimators',100)
        elif model == 'XGBoost':
            with mlflow.start_run(run_name='XGBoost with Fraud'):
                xg_model=XGBClassifier(max_depth=100,n_estimators=10,learning_rate=0.1)
                X=data.drop(['Unnamed: 0','class'],axis=1)
                y=data['class']
                X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
                xg_model.fit(X_train,y_train)
                y_pred=xg_model.predict(X_test)
                accuracy=round(accuracy_score(y_pred,y_test),6)
                precision=round(precision_score(y_pred,y_test),6)
                recall=round(recall_score(y_pred,y_test),6)
                f1=round(f1_score(y_pred,y_test),6)
                print(f"The following metrics are found when the fraud dataset is trained with {model} model")
                print('------------------------')
                print(f"The accuracy is {accuracy}")
                print('------------------------')
                print(f"The precision is {precision}")
                print('------------------------')
                print(f"The recall is {recall}")
                print('------------------------')
                print(f"The f1 score is {f1}")
                print('------------------------')
                #logging the metrics
                mlflow.log_metric('accuracy',accuracy)
                mlflow.log_metric('precision',precision)
                mlflow.log_metric('recall',recall)
                mlflow.log_metric('f1',f1)
                #logging the model itself
                mlflow.sklearn.log_model(xg_model,'xg_model_with_fraud_data')
                #logging the parameters used
                mlflow.log_param('max_depth',100)
                mlflow.log_param('n_estimators',10)
                mlflow.log_param('learning_rate',0.1)
        else:
            print('Please enter a valid machine learning model name')
    elif data == 'creditdata':
        data=pd.read_csv('data/credit_transformed.csv')
        mlflow.set_experiment("Fraud Detection with Credit Dataset")
        if model == 'LogisticRegression':
            with mlflow.start_run(run_name='Logistic Regression with Credit'):
                lr_model= LogisticRegression(C=0.1,max_iter=10,solver='saga')
                X=data.drop(['Unnamed: 0','Class'],axis=1)
                y=data['Class']
                X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
                lr_model.fit(X_train,y_train)
                y_pred=lr_model.predict(X_test)
                accuracy=round(accuracy_score(y_pred,y_test),6)
                precision=round(precision_score(y_pred,y_test),6)
                recall=round(recall_score(y_pred,y_test),6)
                f1=round(f1_score(y_pred,y_test),6)
                print(f"The following metrics are found when the credit dataset is trained with {model} model")
                print('------------------------')
                print(f"The accuracy is {accuracy}")
                print('------------------------')
                print(f"The precision is {precision}")
                print('------------------------')
                print(f"The recall is {recall}")
                print('------------------------')
                print(f"The f1 score is {f1}")
                print('------------------------')
                #logging the metrics
                mlflow.log_metric('accuracy',accuracy)
                mlflow.log_metric('precision',precision)
                mlflow.log_metric('recall',recall)
                mlflow.log_metric('f1',f1)
                #logging the model itself
                mlflow.sklearn.log_model(lr_model,'lr_model_with_credit_data')
                #logging the parameters used
                mlflow.log_param('C',0.1)
                mlflow.log_param('max_iter',10)
                mlflow.log_param('solver','saga')
        elif model == 'DecisionTree':
            with mlflow.start_run(run_name='Decision Tree with Credit'):
                dt_model= DecisionTreeClassifier(max_depth=10)
                X=data.drop(['Unnamed: 0','Class'],axis=1)
                y=data['Class']
                X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
                dt_model.fit(X_train,y_train)
                y_pred=dt_model.predict(X_test)
                accuracy=round(accuracy_score(y_pred,y_test),6)
                precision=round(precision_score(y_pred,y_test),6)
                recall=round(recall_score(y_pred,y_test),6)
                f1=round(f1_score(y_pred,y_test),6)
                print(f"The following metrics are found when the credit dataset is trained with {model} model")
                print('------------------------')
                print(f"The accuracy is {accuracy}")
                print('------------------------')
                print(f"The precision is {precision}")
                print('------------------------')
                print(f"The recall is {recall}")
                print('------------------------')
                print(f"The f1 score is {f1}")
                print('------------------------')
                #logging the metrics
                mlflow.log_metric('accuracy',accuracy)
                mlflow.log_metric('precision',precision)
                mlflow.log_metric('recall',recall)
                mlflow.log_metric('f1',f1)
                #logging the model itself
                mlflow.sklearn.log_model(dt_model,'dt_model_with_credit_data')
                #logging the parameters used
                mlflow.log_param('max_depth',10)
        elif model == 'RandomForest':
            with mlflow.start_run(run_name='Random Forest with Credit'):
                rf_model= RandomForestClassifier(max_depth=10,n_estimators=15)
                X=data.drop(['Unnamed: 0','Class'],axis=1)
                y=data['Class']
                X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
                rf_model.fit(X_train,y_train)
                y_pred=rf_model.predict(X_test)
                accuracy=round(accuracy_score(y_pred,y_test),6)
                precision=round(precision_score(y_pred,y_test),6)
                recall=round(recall_score(y_pred,y_test),6)
                f1=round(f1_score(y_pred,y_test),6)
                print(f"The following metrics are found when the credit dataset is trained with {model} model")
                print('------------------------')
                print(f"The accuracy is {accuracy}")
                print('------------------------')
                print(f"The precision is {precision}")
                print('------------------------')
                print(f"The recall is {recall}")
                print('------------------------')
                print(f"The f1 score is {f1}")
                print('------------------------')
                #logging the metrics
                mlflow.log_metric('accuracy',accuracy)
                mlflow.log_metric('precision',precision)
                mlflow.log_metric('recall',recall)
                mlflow.log_metric('f1',f1)
                #logging the model itself
                mlflow.sklearn.log_model(rf_model,'rf_model_with_credit_data')
                #logging the parameters used
                mlflow.log_param('max_depth',10)
                mlflow.log_param('n_estimators',15)
        elif model == 'XGBoost':
            with mlflow.start_run(run_name='XGBoost with Credit'):
                xg_model= XGBClassifier(max_depth=10,n_estimators=100,learning_rate=1)
                X=data.drop(['Unnamed: 0','Class'],axis=1)
                y=data['Class']
                X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
                xg_model.fit(X_train,y_train)
                y_pred=xg_model.predict(X_test)
                accuracy=round(accuracy_score(y_pred,y_test),6)
                precision=round(precision_score(y_pred,y_test),6)
                recall=round(recall_score(y_pred,y_test),6)
                f1=round(f1_score(y_pred,y_test),6)
                print(f"The following metrics are found when the credit dataset is trained with {model} model")
                print('------------------------')
                print(f"The accuracy is {accuracy}")
                print('------------------------')
                print(f"The precision is {precision}")
                print('------------------------')
                print(f"The recall is {recall}")
                print('------------------------')
                print(f"The f1 score is {f1}")
                print('------------------------')
                #logging the metrics
                mlflow.log_metric('accuracy',accuracy)
                mlflow.log_metric('precision',precision)
                mlflow.log_metric('recall',recall)
                mlflow.log_metric('f1',f1)
                #logging the model itself
                mlflow.sklearn.log_model(xg_model,'xg_model_with_credit_data')
                #logging the parameters used
                mlflow.log_param('max_depth',10)
                mlflow.log_param('n_estimators',100)
                mlflow.log_param('learning_rate',1)
        else:
            print('Please enter a valid machine learning model name')
    else:
        print('Please enter a valid dataset name')



    
    







