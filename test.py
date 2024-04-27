import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Confussion matrix
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import numpy as np

df = pd.DataFrame
def load_csv_variables():
    global variables
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            with open(file_path, 'r') as file:
                global df 
                df = pd.read_csv(file_path)               
                variables = list(df.columns)
                input_variable_listbox.delete(0, tk.END)
                input_variable_listbox.insert(tk.END, *variables)
                target_variable_combobox['values'] = list(df.columns)
              #  print(df.info())
               # return df
        except Exception as e:
            status_label.config(text=f"Error loading CSV file: {e}", fg="red")

def add_input_variable():
    selected_indices = input_variable_listbox.curselection()
    for idx in selected_indices[::-1]:  
        variable = input_variable_listbox.get(idx)
        input_variable_listbox.delete(idx)
        selected_input_variables_listbox.insert(tk.END, variable)
        selected_input_variables.append(variable)

def remove_input_variable():
    selected_indices = selected_input_variables_listbox.curselection()
    for idx in selected_indices[::-1]: 
        variable = selected_input_variables_listbox.get(idx)
        selected_input_variables_listbox.delete(idx)
        input_variable_listbox.insert(tk.END, variable)
        selected_input_variables.remove(variable)
def get_model():
    selected_value = model_combobox.get()
    return selected_value
def get_taget():
    selected_value = target_variable_combobox.get()
    return selected_value

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    return fig
def show_new_page(): 
    global df
    print(df.info())
    new_window = tk.Toplevel(root)  
    new_window.geometry("400x300")
    
    categories=[]
    nums=[]
    selected_variables = selected_input_variables_listbox.get(0, tk.END)
    for variable in selected_variables:
        if(df[variable].dtypes == "object"):
            categories.append(variable)
        else:
            nums.append(variable)
    df_numericals = df[nums]
    imputer = SimpleImputer(strategy='most_frequent')
    df_categories = df[categories]
    df_categories = pd.DataFrame(imputer.fit_transform(df_categories), columns=df_categories.columns)
    print(df_categories.info())
    for cate in categories:
        lb_encoder = LabelEncoder()
        df_categories[cate] = lb_encoder.fit_transform(df[cate])
    
    for num in nums:
        mean = df_numericals[num].mean()
        df_numericals.fillna(mean, inplace=True) 
    X = pd.concat([df_numericals, df_categories], axis=1)
    taget = get_taget()
    if(df[taget].dtype=="object"):
        lb_encoder = LabelEncoder()
        y = lb_encoder.fit_transform(df[taget])
    else:
        y=df[taget]
    print(X.info()) 
    trainX, testX, trainy, testy = train_test_split(X, y , test_size=0.2, random_state=0)
    scaler = StandardScaler()
    trainX = scaler.fit_transform(trainX)
    testX = scaler.transform(testX)

    print("Số lượng mẫu trong tập huấn luyện:", trainX.shape[0])
    print("Số lượng mẫu trong tập kiểm tra:", testX.shape[0])
    model = get_model()
    if(model=="Linear Regression"):
        new_window.title("Linear Regression") 
        model = LinearRegression()
        model.fit(trainX, trainy)
        y_pred = model.predict(testX)
        rmse = mean_squared_error(testy, y_pred, squared=False)
        label = ttk.Label(new_window, text=f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        label.pack(pady=20)
        
    if(model=="Logistic Regression"):
        new_window.title("Logistic Regression") 
        clf = LogisticRegression()
        clf.fit(trainX, trainy)
        predict = clf.predict(testX)
        print(predict)
        
        accuracy = clf.score(testX, testy)
        print("Do chinh xac: ", accuracy)

        label = ttk.Label(new_window, text=f"Độ chính xác: {accuracy:.2f}")
        label.pack(pady=20)
        

        report = classification_report(testy, predict)
        print(report)
        cm = confusion_matrix(testy, predict)
        fig = plot_confusion_matrix(cm, classes=[0, 1])
        
        canvas = FigureCanvasTkAgg(fig, master=new_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        
    if(model=="KNN"):
        new_window.title("KNN")
        k = 5 
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(trainX, trainy)
        y_pred = knn.predict(testX)
        accuracy = accuracy_score(testy, y_pred)
        label = ttk.Label(new_window, text=f"Độ chính xác: {accuracy:.2f}")
        label.pack(pady=20)

variables = []
selected_input_variables = []

root = tk.Tk()
root.title("CSV Variable Selector")
root.geometry("500x500")  

frame = ttk.Frame(root, padding="20")
frame.pack(fill=tk.BOTH, expand=True)

load_button = ttk.Button(frame, text="Load CSV File", command=load_csv_variables)
load_button.pack(pady=10)

target_label = ttk.Label(frame, text="Select Target Variable:")
target_label.pack(pady=5)
target_variable_combobox = ttk.Combobox(frame, state="readonly")
target_variable_combobox.pack(pady=5)

input_label = ttk.Label(frame, text="Select Input Variables:")
input_label.pack(pady=5)

input_variable_listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, height=3)
input_variable_listbox.pack(pady=5)

add_button = ttk.Button(frame, text="Add", command=add_input_variable)
add_button.pack(pady=5)

selected_label = ttk.Label(frame, text="Selected Input Variables:")
selected_label.pack(pady=5)

selected_input_variables_listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, height=3)
selected_input_variables_listbox.pack(pady=5)

remove_button = ttk.Button(frame, text="Remove", command=remove_input_variable)
remove_button.pack(pady=5)

status_label = ttk.Label(frame, text="Chọn Model:")
status_label.pack(pady=10)
model_combobox = ttk.Combobox(frame, values=["Linear Regression", "Logistic Regression", "KNN"], state="readonly")
model_combobox.pack(pady=5)

add_button = ttk.Button(frame, text="Excution", command=show_new_page)
add_button.pack(pady=5)

root.mainloop()
