from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
import os
import matplotlib.pyplot as plt
import joblib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from tkinter import Tk, Label, Button, Entry 
from PIL import Image, ImageTk
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor

cls_names = []
Accuracy_list = []
Precision_list = []
Recall_list = []
F1_list = []
AUC_list = []

reg_names = []
R2_list = []
MAE_list = []
RMSE_list = []


global filename
model_folder = "model"

global X_train,X_test,y_class_train,y_class_test,y_reg_train,y_reg_test

scaler = StandardScaler()

#UPLOADING THE DATASET-------------------------------------
def uploadDataset(): 
    global df  
    filename = filedialog.askopenfilename(initialdir = "df")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n')
    df = pd.read_csv(filename)
    text.insert(END,str(df.head())+"\n\n")

#DATA PREPROCESSING------------------------
def preprocess_and_scale_features(df, target_cols=["response_efficiency_score", "recovery_days"]):
    df = df.copy()

    cat_cols = ["country", "disaster_type"]
    num_cols = df.select_dtypes(include=["int32", "int64", "float64"]).columns
    num_cols = [c for c in num_cols if c not in target_cols]

    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df, scaler, label_encoders


def Preprocess_Dataset():
    global dataset,df, scaler, label_encoders
    global X,y
    global X,y_class,y_reg
    
    text.delete('1.0', END)
    
    LE =  LabelEncoder()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df.drop("date", axis=1, inplace=True)
    
    # Convert to classification categories
    df['response_efficiency_score'] = pd.qcut(
        df['response_efficiency_score'],
        q=3,
        labels=["Low", "Medium", "High"]
    )
    
    df, scaler, label_encoders = preprocess_and_scale_features(df)
    
    X = df.drop(["response_efficiency_score", "recovery_days"], axis=1)
    y_class = df['response_efficiency_score']
    y_reg = df['recovery_days']
    text.insert(END,str(X.head())+"\n\n")

    
#EDA--------------------------------------
def disaster_eda_plots(df):
    text.delete('1.0', END)
    
    country_count = df['country'].value_counts().head(10)
    
    plt.figure(figsize=(10,5))
    sns.barplot(x=country_count.index, y=country_count.values)
    plt.xticks(rotation=45)
    plt.title("Top 10 Most Impacted Countries")
    plt.tight_layout()
    plt.show()  
    fig, ax = plt.subplots(figsize=(14,6))
    sns.histplot(df['casualties'], kde=True, ax=ax)   # distplot is deprecated
    ax.set_title("Casualties Distribution")
    plt.tight_layout()
    plt.show()
    text.insert(END, "EDA Process Completed")
    

#TRAIN TEST SPLITTING DATA FOR MODEL TRAINING--------------------------------------
def Train_Test_Splitting(): 
    text.delete('1.0',END)
    global X,y_class,y_reg
    global X_train,X_test,y_class_train,y_class_test,y_reg_train,y_reg_test

    #Splitting the data
    X_train, X_test, y_class_train, y_class_test,y_reg_train,y_reg_test = train_test_split(
        X, y_class,y_reg, test_size=0.2, random_state=42
    )
    text.insert(END, "Total records found in dataset: " + str(X.shape[0]) + "\n\n")
    text.insert(END, "Total records found in dataset to train: " + str(y_reg_train.shape[0]) + "\n\n")
    text.insert(END, "Total records found in dataset to test: " + str(y_class_test.shape[0]) + "\n\n")

def calculate_metrics(
    task_type,
    algorithm,
    y_test,
    preds,
    probs=None
):
    """
    task_type: 'classification' or 'regression'
    """

    #=========CLASSIFICATION==========#
    if task_type=='classification':

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

        # Safe AUC calculation
        auc = np.nan
        try:
            if probs is not None and hasattr(probs, "ndim") and probs.ndim == 2:
                auc = roc_auc_score(y_test, probs, multi_class="ovr")
        except:
            auc = np.nan

        cls_names.append(algorithm)
        Accuracy_list.append(acc)
        Precision_list.append(prec)
        Recall_list.append(rec)
        F1_list.append(f1)
        AUC_list.append(auc)

        # ---- DISPLAY IN GUI ----
        text.insert(END, f"\n{algorithm} Classification Results\n")
        text.insert(END, f"Accuracy  : {acc:.4f}\n")
        text.insert(END, f"Precision : {prec:.4f}\n")
        text.insert(END, f"Recall    : {rec:.4f}\n")
        text.insert(END, f"F1-score  : {f1:.4f}\n")

        if not np.isnan(auc):
            text.insert(END, f"AUC       : {auc:.4f}\n")

        text.insert(END, "-" * 40 + "\n")


    #===========REGRESSION=========#
    elif task_type == "regression":

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        reg_names.append(algorithm)
        R2_list.append(r2)
        MAE_list.append(mae)
        RMSE_list.append(rmse)

        text.insert(END, f"\n{algorithm} Regression Results:\n")
        text.insert(END, f"R2 Score : {r2:.4f}\n")
        text.insert(END, f"MAE      : {mae:.4f}\n")
        text.insert(END, f"RMSE     : {rmse:.4f}\n")
        text.insert(END, "-"*40 + "\n")

        os.makedirs("results", exist_ok=True)

        plt.figure(figsize=(6, 5))
        plt.scatter(y_test, preds, alpha=0.6)
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            linestyle="--"
        )
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{algorithm}: Actual vs Predicted")
        plt.tight_layout()

        plt.savefig(
            f"results/{algorithm.replace(' ','_').lower()}_actual_vs_pred.png"
        )

        plt.show()



def plot_confusion_matrix(
    y_true,
    y_pred,
    class_labels,
    title="Confusion Matrix"
):
    cm=confusion_matrix(y_true,y_pred,labels=class_labels)

    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels
    )
    
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_regression_comparison():
    if len(reg_names) == 0:
        print("No regression metrics available")
        return
    data_len = min(
    len(reg_names),
    len(R2_list),
    len(MAE_list),
    len(RMSE_list)
    )
    
    x = np.arange(data_len)
    w = 0.25
    
    plt.figure(figsize=(14,6))
    plt.bar(x - w, R2_list[:data_len], w, label="R2")
    plt.bar(x, MAE_list[:data_len], w, label="MAE")
    plt.bar(x + w, RMSE_list[:data_len], w, label="RMSE")
    
    plt.xticks(x, reg_names[:data_len], rotation=20)
    plt.ylabel("Score")
    plt.title("Regression Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()


#MODELS/ALGORITHMS
#Decision Tree Model------------------------------
def decision_tree_model(
    X_train, X_test,
    y_class_train, y_class_test,
    y_reg_train, y_reg_test,
    folder="models",
    max_depth=5,
    random_state=42,
    class_labels=("Low", "Medium", "High")
):
    

    os.makedirs(folder, exist_ok=True)

    # ============== CLASSIFICATION ==============
    clf_path = f"{folder}/decision_tree_classifier.pkl"

    if os.path.exists(clf_path):
        dt_clf = joblib.load(clf_path)
    else:
        dt_clf = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state
        )
        dt_clf.fit(X_train, y_class_train)
        joblib.dump(dt_clf, clf_path)

    clf_preds = dt_clf.predict(X_test)
    
    # Safe probability handling
    clf_probs = (
        dt_clf.predict_proba(X_test)
        if hasattr(dt_clf, "predict_proba")
        else None
    )
    
    calculate_metrics(
        task_type="classification",
        algorithm="Decision Tree",
        y_test=y_class_test,
        preds=clf_preds,
        probs=clf_probs
    )
    plot_confusion_matrix(
        y_class_test,
        clf_preds,
        class_labels,
        title="Decision Tree 🔲 Confusion Matrix"
    )
    
    # =================== REGRESSION ===================
    reg_path = f"{folder}/decision_tree_regressor.pkl"
    
    if os.path.exists(reg_path):
        dt_reg = joblib.load(reg_path)
    else:
        dt_reg = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=random_state
        )
        dt_reg.fit(X_train, y_reg_train)
        joblib.dump(dt_reg, reg_path)
    
    reg_preds = dt_reg.predict(X_test)
    
    calculate_metrics(
        task_type="regression",
        algorithm="Decision Tree",
        y_test=y_reg_test,
        preds=reg_preds
    )

    return dt_clf,dt_reg

#AdaBoost Model--------------------------------
def adaboost_model(
    X_train, X_test,
    y_class_train, y_class_test,
    y_reg_train, y_reg_test,
    folder="models",
    random_state=42,
    class_labels=("Low", "Medium", "High")
):

    os.makedirs(folder, exist_ok=True)

    # =============== CLASSIFICATION ===============
    clf_path = f"{folder}/adaboost_classifier.pkl"

    if os.path.exists(clf_path):
        ada_clf = joblib.load(clf_path)
    else:
        ada_clf = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=100,
            learning_rate=0.5,
            random_state=random_state
        )

        ada_clf.fit(X_train, y_class_train)
        joblib.dump(ada_clf, clf_path)

    clf_preds = ada_clf.predict(X_test)
    clf_probs = ada_clf.predict_proba(X_test)

    calculate_metrics(
        task_type="classification",
        algorithm="AdaBoost",
        y_test=y_class_test,
        preds=clf_preds,
        probs=clf_probs
    )

    plot_confusion_matrix(
        y_class_test,
        clf_preds,
        class_labels,
        title="AdaBoost 🔲 Confusion Matrix"
    )

    # ============== REGRESSION ==============
    reg_path = f"{folder}/adaboost_regressor.pkl"

    if os.path.exists(reg_path):
        ada_reg = joblib.load(reg_path)
    else:
        ada_reg = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(),
            n_estimators=100,
            learning_rate=0.5,
            random_state=random_state
        )
        ada_reg.fit(X_train, y_reg_train)
        joblib.dump(ada_reg, reg_path)

    reg_preds = ada_reg.predict(X_test)

    calculate_metrics(
        task_type="regression",
        algorithm="AdaBoost",
        y_test=y_reg_test,
        preds=reg_preds
    )

    return ada_clf, ada_reg

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor


#SGD model------------------------------------
def sgd_model(
    X_train, X_test,
    y_class_train, y_class_test,
    y_reg_train, y_reg_test,
    folder="models",
    random_state=42,
    class_labels=("Low", "Medium", "High")
):

    os.makedirs(folder, exist_ok=True)

    # -------- Classification --------
    clf_path = f"{folder}/sgd_classifier.pkl"

    if os.path.exists(clf_path):
        sgd_clf = joblib.load(clf_path)
    else:
        sgd_clf = SGDClassifier(
            loss="log_loss",
            alpha=0.1,
            max_iter=1000,
            random_state=random_state
        )

        sgd_clf.fit(X_train, y_class_train)
        joblib.dump(sgd_clf, clf_path)
        
    clf_preds = sgd_clf.predict(X_test)
    clf_probs = sgd_clf.predict_proba(X_test)
    
    calculate_metrics(
        "classification", "SGD",
        y_class_test, clf_preds, clf_probs
    )
    
    plot_confusion_matrix(
        y_class_test,
        clf_preds,
        class_labels,
        title="SGD 🤖 Confusion Matrix"
    )
    
    # -------- Regression --------
    reg_path = f"{folder}/sgd_regressor.pkl"
    
    if os.path.exists(reg_path):
        sgd_reg = joblib.load(reg_path)
    else:
        sgd_reg = SGDRegressor(
            loss="squared_error",
            alpha=0.5,
            max_iter=1000,
            random_state=random_state
        )
        sgd_reg.fit(X_train, y_reg_train)
        joblib.dump(sgd_reg, reg_path)
    
    reg_preds = sgd_reg.predict(X_test)
    
    calculate_metrics(
        "regression", "SGD",
        y_reg_test, reg_preds
    )
    
    return sgd_clf, sgd_reg

from catboost import CatBoostClassifier
from catboost import CatBoostRegressor

#CatBoost Model------------------------
def catboost_model(
    X_train, X_test,
    y_class_train, y_class_test,
    y_reg_train, y_reg_test,
    folder="models",
    random_state=42,
    class_labels=("Low", "Medium", "High")
):
    global cat_clf, cat_reg
    os.makedirs(folder, exist_ok=True)

    # ================= CLASSIFICATION =================
    clf_path = f"{folder}/catboost_classifier.pkl"

    if os.path.exists(clf_path):
        cat_clf = joblib.load(clf_path)
    else:
        cat_clf = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_state=random_state,
            verbose=0
        )

        cat_clf.fit(X_train, y_class_train)
        joblib.dump(cat_clf, clf_path)

    clf_preds = cat_clf.predict(X_test)
    clf_probs = cat_clf.predict_proba(X_test)

    calculate_metrics(
        task_type="classification",
        algorithm="CatBoost",
        y_test=y_class_test,
        preds=clf_preds,
        probs=clf_probs
    )

    plot_confusion_matrix(
        y_class_test,
        clf_preds,
        class_labels,
        title="CatBoost 📊 Confusion Matrix"
    )

    # ================ REGRESSION ================
    reg_path = f"{folder}/catboost_regssion.pkl"

    if os.path.exists(reg_path):
        cat_reg = joblib.load(reg_path)
    else:
        cat_reg = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_state=random_state,
            verbose=0
        )
        cat_reg.fit(X_train, y_reg_train)
        joblib.dump(cat_reg, reg_path)

    reg_preds = cat_reg.predict(X_test)

    calculate_metrics(
        task_type="regression",
        algorithm="CatBoost",
        y_test=y_reg_test,
        preds=reg_preds
    )

    return cat_clf, cat_reg


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_classification_comparison():
    if not cls_names:
        print("No classification results available.")
        return

    # Prepare DataFrame
    data = []
    for i, model in enumerate(cls_names):
        data.append([model, "Accuracy", Accuracy_list[i]])
        data.append([model, "Precision", Precision_list[i]])
        data.append([model, "Recall", Recall_list[i]])
        data.append([model, "F1-score", F1_list[i]])
        data.append([model, "AUC", AUC_list[i] if not np.isnan(AUC_list[i]) else 0])

    df_plot = pd.DataFrame(data, columns=["Model", "Metric", "Value"])
    pivot_df = df_plot.pivot(index="Metric", columns="Model", values="Value")

    # Plot
    pivot_df.plot(kind="bar", figsize=(10,6))
    plt.title("Classification Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.show()


def plot_regression_comparison_graph():
    if not reg_names:
        print("No regression results available.")
        return

    # Prepare DataFrame
    data = []
    for i, model in enumerate(reg_names):
        data.append([model, "R2 Score", R2_list[i]])
        data.append([model, "MAE", MAE_list[i]])
        data.append([model, "RMSE", RMSE_list[i]])

    df_plot = pd.DataFrame(data, columns=["Model", "Metric", "Value"])
    pivot_df = df_plot.pivot(index="Metric", columns="Model", values="Value")

    # Plot
    pivot_df.plot(kind="bar", figsize=(10,6))
    plt.title("Regression Model Performance Comparison")
    plt.ylabel("Score / Error")
    plt.xticks(rotation=0)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.show()



#PREDICTION---------------------------------------------------
def Prediction():
    global Testdata, cat_clf, cat_reg, scaler, label_encoders, LE
    
    filename = filedialog.askopenfilename(initialdir="Testdata")
    text.delete('1.0', END)
    text.insert(END, filename + ' Loaded\n\n')
    
    Testdata = pd.read_csv(filename)

    # ---------------- Date Handling ----------------
    Testdata["date"] = pd.to_datetime(Testdata["date"], errors="coerce")
    Testdata["year"] = Testdata["date"].dt.year
    Testdata["month"] = Testdata["date"].dt.month
    Testdata.drop("date", axis=1, inplace=True)

    # ---------------- Encoding ----------------
    for col in Testdata.columns:
        if Testdata[col].dtype == "object":
            if col in label_encoders:
                Testdata[col] = label_encoders[col].transform(Testdata[col])
            else:
                Testdata[col] = LE.fit_transform(Testdata[col])

    # ---------------- Scaling ----------------
    num_cols = scaler.feature_names_in_
    Testdata[num_cols] = scaler.transform(Testdata[num_cols])

    # ---------------- Prediction ----------------
    pred_class = cat_clf.predict(Testdata).ravel()
    pred_reg = cat_reg.predict(Testdata).ravel()

    Testdata["Pred_Response_Efficiency"] = pred_class
    Testdata["Pred_Recovery_Days"] = pred_reg

    
    text.insert(END, "ROW WISE PREDICTION RESULTS:\n")
    text.insert(END, "-"*120 + "\n\n")

    for i, row in Testdata.iterrows():

        # ----- Row Data Line -----
        row_output = f"Row {i+1} : "

        for col in Testdata.columns[:-2]:  
            value = row[col]
            row_output += f"{col}={value} | "

        text.insert(END, row_output + "\n")

        # ----- Prediction Line -----
        pred_output = f"Prediction → Response_Efficiency={row['Pred_Response_Efficiency']} | Recovery_Days={row['Pred_Recovery_Days']}"
        text.insert(END, pred_output + "\n\n")  


def close():
    main.destroy()

main=Tk()

# Create main window
main.title("Machine Learning Based Assessment of Disaster Response Efficiency and Regional Recovery Dynamics")
# Get the screen width and height
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()

# Set the window size to the screen dimensions
main.geometry(f"{screen_width}x{screen_height}")

# ============== BACKGROUND IMAGE ===============================
bg_image = Image.open("response.jpg").resize((1400, 900))
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = Label(main, image=bg_photo)
bg_label.image=bg_photo 
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Send background to back
bg_label.lower()
#-------------------------------------------

font = ('times', 16, 'bold')
title = Label(main, text='Machine Learning Based Assessment of Disaster Response Efficiency and Regional Recovery Dynamics')
title.config(bg='black', fg='white')  
title.config(font=font)           
title.config(height=3, width=130)       
title.place(x=0,y=5)
font1 = ('times', 15, 'bold')
ff = ('times', 15, 'bold')

#BUTTONS-----------------------------------------------------------
uploadButton = Button(main, text="Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=ff)
uploadButton.config(bg='lightblue',fg='red')

preprocessButton = Button(main, text="Preprocessing", command=Preprocess_Dataset)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=ff)
preprocessButton.config(bg='lightblue',fg='red')

edaButton = Button(main, text="EDA", command=lambda:disaster_eda_plots(df))
edaButton.place(x=50,y=200)
edaButton.config(font=ff)
edaButton.config(bg='lightblue',fg='red')

TrainTestSplittingButton= Button(main, text="Train Test Splitting", command=Train_Test_Splitting)
TrainTestSplittingButton.place(x=50,y=250)
TrainTestSplittingButton.config(font=ff)
TrainTestSplittingButton.config(bg='lightblue',fg='red')

DecisionTreeButton = Button(main, text="Decision Tree model", command= lambda: decision_tree_model(
    X_train, X_test,
    y_class_train, y_class_test,
    y_reg_train, y_reg_test,
    folder="models",
    max_depth=5,
    random_state=42,
    class_labels=("Low", "Medium", "High")
))
DecisionTreeButton.place(x=50,y=300)
DecisionTreeButton.config(font=ff)
DecisionTreeButton.config(bg='lightblue',fg='red')

AdaBoostButton = Button(main, text="AdaBoost model", command= lambda: adaboost_model(
    X_train, X_test,
    y_class_train, y_class_test,
    y_reg_train, y_reg_test,
    folder="models",
    random_state=42,
    class_labels=("Low", "Medium", "High")
))
AdaBoostButton.place(x=50,y=350)
AdaBoostButton.config(font=ff)
AdaBoostButton.config(bg='lightblue',fg='red')

sgdButton = Button(main, text="SGD model", command= lambda: sgd_model(
    X_train, X_test,
    y_class_train, y_class_test,
    y_reg_train, y_reg_test,
    folder="models",
    random_state=42,
    class_labels=("Low", "Medium", "High")
))
sgdButton.place(x=50,y=400)
sgdButton.config(font=ff)
sgdButton.config(bg='lightblue',fg='red')

CatBoostButton = Button(main, text="CatBoost model", command= lambda: catboost_model(
    X_train, X_test,
    y_class_train, y_class_test,
    y_reg_train, y_reg_test,
    folder="models",
    random_state=42,
    class_labels=("Low", "Medium", "High")
))
CatBoostButton.place(x=50,y=450)
CatBoostButton.config(font=ff)
CatBoostButton.config(bg='lightblue',fg='red')

classification_comparison= Button(main, text="Classification Comparison", command=plot_classification_comparison)
classification_comparison.place(x=50,y=500)
classification_comparison.config(font=ff)
classification_comparison.config(bg='lightblue',fg='red')

regression_comparison= Button(main, text="Regression Comparison", command=plot_regression_comparison_graph)
regression_comparison.place(x=50,y=550)
regression_comparison.config(font=ff)
regression_comparison.config(bg='lightblue',fg='red')

PredictionButton = Button(main, text="Prediction", command=Prediction)
PredictionButton.place(x=50,y=600)
PredictionButton.config(font=ff)
PredictionButton.config(bg='lightblue',fg='red')

Button1 = Button(main, text="Exit", command=close, bg='Red', fg='black')
Button1.place(x=20,y=650)
Button1.config(font=ff)

#--------------------------------------------------------------------

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=70)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=350,y=120)
text.config(font=font1)

main.config(bg='orange')
main.mainloop()

