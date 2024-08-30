import mlflow 
import mlflow.sklearn
from sklearn.datasets import load_wine 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns 
import dagshub 

# **`mlflow.autolog()`** is a powerful feature in MLflow that automatically logs parameters, metrics, models, and other relevant information during your machine learning training process. However, it's important to know what can and cannot be logged automatically.

# ### **Things That Can Be Logged by `mlflow.autolog`:**

# 1. **Parameters:**
#    - Hyperparameters used to train the model, such as `max_depth`, `learning_rate`, `n_estimators`, etc.

# 2. **Metrics:**
#    - Common evaluation metrics like accuracy, precision, recall, and loss values, depending on the model and framework being used.

# 3. **Model:**
#    - The trained model itself is automatically logged.

# 4. **Artifacts:**
#    - Certain artifacts like model summary and plots (e.g., learning curves, confusion matrix) are logged if supported by the framework.

# 5. **Framework-Specific Information:**
#    - Framework-specific details like early stopping criteria in gradient boosting models or deep learning models (e.g., number of epochs, optimizer configuration).

# 6. **Environment Information:**
#    - Details about the environment such as installed libraries and versions.

# 7. **Training Data and Labels:**
#    - Information about the dataset size and sometimes feature information, but not the entire dataset itself.

# 8. **Automatic Model Signature:**
#    - Autologging can infer the input types (signature) of the model and save them along with the model.

# ### **Things That Cannot Be Logged by `mlflow.autolog`:**

# 1. **Custom Metrics:**
#    - Metrics not included in the default set for the specific framework (e.g., F1 score if it's not the default metric) will not be logged unless manually specified.

# 2. **Custom Artifacts:**
#    - Custom plots, charts, or files that are not part of the default model training process (e.g., a custom visualization or report).

# 3. **Preprocessed Data:**
#    - The transformed or preprocessed data used during training or testing is not logged unless you manually log it as an artifact.

# 4. **Intermediate Model States:**
#    - Models saved at intermediate stages of training (e.g., after every epoch) are not logged unless explicitly done so.

# 5. **Complex Model Structures:**
#    - If you're using a non-standard or highly customized model structure, `mlflow.autolog` might miss some logging details.

# 6. **Non-standard Training Loops:**
#    - If your training loop is not compatible with the standard loops expected by MLflow (e.g., custom training loops), autologging might not capture everything correctly.

# 7. **Non-Supported Frameworks:**
#    - `mlflow.autolog` does not support all frameworks. If your model is built with a framework that MLflow doesn’t support, autologging won’t work.

# 8. **Custom Hyperparameter Tuning:**
#    - Hyperparameters or configurations that are outside the scope of the framework’s autologging capabilities (e.g., specific settings in a custom grid search).

# ### Summary:

# - **Use Cases:** `mlflow.autolog` is great for quick and convenient logging, especially for standard workflows in supported frameworks.
# - **Limitations:** Custom elements, complex structures, and unsupported frameworks require manual logging to capture all relevant details.

dagshub.init(repo_owner='sudarshansoni2000', repo_name='mlfow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/sudarshansoni2000/mlfow.mlflow")

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

mlflow.autolog()
mlflow.set_experiment("MLOPS-Exp1")


with mlflow.start_run():  #experiment_id= "429411590396887716"

    rf = RandomForestClassifier(max_depth = max_depth , n_estimators = n_estimators , random_state= 42)
    rf.fit(X_train,y_train)

    y_pred=rf.predict(X_test)
    accuracy= accuracy_score(y_test , y_pred)



# creating a confusion matrix plot 
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm , annot=True , fmt="d" , cmap="Blues" , xticklabels=wine.target_names , yticklabels=wine.target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("confusion_matrix")

    # save plot
    plt.savefig("confusion-matrix.png")

     

    mlflow.log_artifact(__file__)


    # tags

    mlflow.set_tags({"Author": "Sudarshan" , "Project": "Wine Classification"})

    print(accuracy) 