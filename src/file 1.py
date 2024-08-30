import mlflow 
import mlflow.sklearn
from sklearn.datasets import load_wine 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns 
import dagshub

# MLflow is a powerful tool for tracking and managing machine learning experiments. Here’s a list of things that can be tracked/logged using MLflow, which you can include in your tutorial documentation:

# ### 1. **Metrics:**
#    - **Accuracy**: Track model accuracy over different runs.
#    - **Loss**: Log training and validation loss during the training process.
#    - **Precision, Recall, F1-Score**: Log evaluation metrics for classification tasks.
#    - **AUC (Area Under Curve)**: Track AUC for classification models.
#    - **Custom Metrics**: Any numeric value can be logged as a custom metric (e.g., RMSE, MAE).

# ### 2. **Parameters:**
#    - **Model Hyperparameters**: Log values such as learning rate, number of trees, max depth, etc.
#    - **Data Processing Parameters**: Track parameters used in data preprocessing, such as the ratio of train-test split or feature selection criteria.
#    - **Feature Engineering**: Log any parameters related to feature extraction or engineering.

# ### 3. **Artifacts:**
#    - **Trained Models**: Save and version models for easy retrieval and comparison.
#    - **Model Summaries**: Log model summaries or architecture details.
#    - **Confusion Matrices**: Save visualizations of confusion matrices.
#    - **ROC Curves**: Log Receiver Operating Characteristic curves.
#    - **Plots**: Save any custom plots like loss curves, feature importances, etc.
#    - **Input Data**: Log the datasets used in training and testing.
#    - **Scripts & Notebooks**: Save code files or Jupyter notebooks used in the experiment.
#    - **Environment Files**: Track environment files like `requirements.txt` or `conda.yaml` to ensure reproducibility.

# ### 4. **Models:**
#    - **Pickled Models**: Log models in a serialized format that can be reloaded later.
#    - **ONNX Models**: Log models in the ONNX format for cross-platform usage.
#    - **Custom Models**: Log custom models using MLflow’s model interface.

# ### 5. **Tags:**
#    - **Run Tags**: Tag your experiments with metadata like author name, experiment description, or model type.
#    - **Environment Tags**: Tag with environment-specific details like `gpu` or `cloud_provider`.

# ### 6. **Source Code:**
#    - **Scripts**: Track the script or notebook used in the experiment.
#    - **Git Commit**: Log the Git commit hash to link the experiment with a specific version of the code.
#    - **Dependencies**: Track the exact version of libraries and dependencies used.

# ### 7. **Logging Inputs and Outputs:**
#    - **Training Data**: Log the training data used in the experiment.
#    - **Test Data**: Log the test or validation datasets.
#    - **Inference Outputs**: Track the predictions or outputs of the model on a test set.

# ### 8. **Custom Logging:**
#    - **Custom Objects**: Log any Python object or file type as a custom artifact.
#    - **Custom Functions**: Track custom functions or methods used within the experiment.

# ### 9. **Model Registry:**
#    - **Model Versioning**: Track different versions of models and their lifecycle stages (e.g., `Staging`, `Production`).
#    - **Model Deployment**: Manage and track the deployment status of different models.

# ### 10. **Run and Experiment Details:**
#    - **Run ID**: Each run is assigned a unique identifier.
#    - **Experiment Name**: Group multiple runs under a single experiment name.
#    - **Timestamps**: Log start and end times of each run to track duration.

# This comprehensive list should help users understand the versatility of MLflow for tracking and managing their machine learning workflows.

mlflow.set_tracking_uri("http://127.0.0.1:5000")

## Load wine dataset 
wine = load_wine()
X= wine.data
y= wine.target


# train_test split
X_train , X_test , y_train , y_test = train_test_split(X,y, test_size=0.10 , random_state= 42)

# Define the prams for random forest 
max_depth = 40
n_estimators = 30

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

    mlflow.set_tags({"Author": "Sudarshan" , "Project": "Wine Classification"})

    # log the model 
    mlflow.sklearn.log_model(rf , "Random Forest classifier model")
    print(accuracy) 