# using nter
import csv
from tkinter import *
import tkinter as tk
from tkinter import filedialog, ttk

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

csv_columns = []
csv_dt = []


def load_csv_file():
    global csv_columns
    global csv_dt
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        with open(file_path, "r") as file:
            csv_data = csv.reader(file)
            csv_dt = pd.read_csv(file_path)
            csv_columns = next(csv_data)
        target_combobox["values"] = csv_columns
      
        update_input_listbox()


def update_input_listbox():
    input_listbox.delete(0, END)
    for column in csv_columns:
        if column != target_combobox.get():
            input_listbox.insert(END, column)


def add_variable():
 
    selected_indices = input_listbox.curselection()
    for index in selected_indices:
        selected_variable = input_listbox.get(index)
        selected_listbox.insert(END, selected_variable)
        input_listbox.delete(index)
    input_listbox.selection_clear(0, END)


def remove_variable():
    selected_indices = selected_listbox.curselection()
    for index in selected_indices:
        removed_variable = selected_listbox.get(index)
        input_listbox.insert(END, removed_variable)
        selected_listbox.delete(index)
    selected_listbox.selection_clear(0, END)


def execute_model():
    target_variable = target_combobox.get()
    input_variables = selected_listbox.get(0, END)
    input_variables = list(input_variables)
    model = model_combobox.get()

    X = csv_dt[input_variables]
    y = csv_dt[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    global model_train
    if model == "Logistic Regression":
        model_train = LogisticRegression()
    elif model == "KNN":
        model_train = KNeighborsClassifier()

    elif model == "Linear Regression":
        model_train = LinearRegression()

    model_train.fit(X_train, y_train)
    y_pred = model_train.predict(X_test)

    if isinstance(model_train, LogisticRegression) or isinstance(
        model_train, KNeighborsClassifier
    ):
        try:
            accuracy = accuracy_score(y_test, y_pred)

            res.configure(text=f"Accuracy : {accuracy:.2f}")

        except:
            # res.configure(text=f"Accuracy : {0.00}")
            print("error")

    elif isinstance(model_train, LinearRegression):
        r2 = str(r2_score(y_test, y_pred))
        res.configure(text=f"Accuracy : {r2.format('.2f')}")

def load_table():
    print(csv_columns)


root = Tk()
width = 900
height = 700


# slider
frame = Frame(root, bg="#eee", width=300, height=height)
frame.grid(row=0, column=0, sticky=N + S + E + W)


load_button = Button(frame, text="Load CSV File", command=load_csv_file, width=25)
load_button.grid(row=0, column=0, padx=5, pady=5, sticky=W)

target_label = Label(frame, text="Select Target Variable", width=25)
target_label.grid(row=1, column=0, padx=5, sticky=W)

target_combobox = ttk.Combobox(frame, width=26)
target_combobox.grid(row=2, column=0, padx=5, pady=5, sticky=W)


input_label = Label(frame, text="Select Input Variables", width=25)
input_label.grid(row=3, column=0, padx=5, sticky=W)


input_scrollbar = Scrollbar(frame)
input_scrollbar.grid(row=4, column=1, sticky=N + S)


input_listbox = Listbox(frame, yscrollcommand=input_scrollbar.set, selectmode=MULTIPLE)
input_listbox.grid(row=4, column=0, padx=5, pady=5, sticky=W + E + N + S)


input_scrollbar.config(command=input_listbox.yview)

# Tạo nút "Add"
add_button = Button(frame, text="Add", command=add_variable, width=25)
add_button.grid(row=5, column=0, padx=5, pady=5, sticky=W)

# Tạo nhãn "Selected Input Variables"
selected_label = Label(frame, text="Selected Input Variables", width=25)
selected_label.grid(row=6, column=0, padx=5, sticky=W)

# Tạo scrollbar cho danh sách các Selected Input Variables
selected_scrollbar = Scrollbar(frame)
selected_scrollbar.grid(row=7, column=1, sticky=N + S)

# Tạo danh sách các Selected Input Variables
selected_listbox = Listbox(frame, yscrollcommand=selected_scrollbar.set, width=25)
selected_listbox.grid(row=7, column=0, padx=5, pady=5, sticky=W + E + N + S)

# Kết nối scrollbar với danh sách các Selected Input Variables
selected_scrollbar.config(command=selected_listbox.yview)

# Tạo nút "Remove"
remove_button = Button(frame, text="Remove", command=remove_variable, width=25)
remove_button.grid(row=8, column=0, padx=5, pady=5, sticky=W)

# Tạo nhãn "Chọn Model"
model_label = Label(frame, text="Chọn Model", width=25)
model_label.grid(row=9, column=0, padx=5, pady=10, sticky=W)

# Tạo combobox để chọn Model
model_combobox = ttk.Combobox(
    frame, values=["Logistic Regression", "KNN", "Linear Regression"], width=26
)
model_combobox.grid(row=10, column=0, padx=5, sticky=W)

# Tạo nút "Execution"
execution_button = Button(frame, text="Execution", command=execute_model, width=25)
execution_button.grid(row=11, column=0, padx=5, pady=10, sticky=W)

# content
content = Frame(root, bg="red", width=900, height=height)
content.grid(row=0, column=1)

temp = Text(content, width=900)
temp.grid(row=1, column=0, sticky=W)

res = Label(content, text="Result", width=25)
res.grid(row=0, column=0, padx=5, pady=5, sticky=W)
        # Giao diện hiển thị thông tin Data từ File CSV
# self.result_text = tk.Text(self.result_frame, height=10, width=80)
# self.result_text.pack(expand=False , side=tk.BOTTOM, fill=tk.BOTH, pady=10)

# selected_scrollbar = Scrollbar(root)
# selected_scrollbar.grid(row=7, column=1, sticky=N + S)

setW = root.winfo_screenwidth()
setH = root.winfo_screenheight()

# chính giữa màn hình
xLeft = (setW / 2) - (width / 2)
yTop = (setH / 2) - (height / 2)
# size window

root.geometry("%dx%d+%d+%d" % (width, height, xLeft, yTop))

root.mainloop()
