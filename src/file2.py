import mlflow 
import mlflow.sklearn
from sklearn.datasets import load_wine 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns 
import dagshub

dagshub.init(repo_owner='sudarshansoni2000', repo_name='mlfow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/sudarshansoni2000/mlfow.mlflow")

## Load wine dataset 
wine = load_wine()
X= wine.data
y= wine.target


# train_test split
X_train , X_test , y_train , y_test = train_test_split(X,y, test_size=0.10 , random_state= 42)

# Define the prams for random forest 
max_depth = 55
n_estimators = 66

# Mention your experiment name 
mlflow.set_experiment("MLOPS-Exp1")


with mlflow.start_run():  #experiment_id= "429411590396887716"

    rf = RandomForestClassifier(max_depth = max_depth , n_estimators = n_estimators , random_state= 42)
    rf.fit(X_train,y_train)

    y_pred=rf.predict(X_test)
    accuracy= accuracy_score(y_test , y_pred)
          
    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_param("max_depth",max_depth)
    mlflow.log_param("n_estimators",n_estimators)


# creating a confusion matrix plot 
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm , annot=True , fmt="d" , cmap="Blues" , xticklabels=wine.target_names , yticklabels=wine.target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("confusion_matrix")

    # save plot
    plt.savefig("confusion-matrix.png")

     
    # log_artifact usnig mlflow 
    mlflow.log_artifact("confusion-matrix.png")
    mlflow.log_artifact(__file__)


    # tags

    mlflow.set_tags({"Author": "Shashwat" , "Project": "Wine Classification"})

    # log the model 
    mlflow.sklearn.log_model(rf , "Random Forest classifier model")
    print(accuracy) 