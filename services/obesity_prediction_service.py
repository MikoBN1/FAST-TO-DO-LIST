# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_data_from_excel():
    return pd.read_excel("datasets/obes_dataset.xlsx")

df = load_data_from_excel()
categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC',
                    'SMOKE', 'SCC', 'CALC', 'MTRANS']

def encode_dataframe():
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    target_encoder = LabelEncoder()
    df['NObeyesdad'] = target_encoder.fit_transform(df['NObeyesdad'])
    numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

encode_dataframe()

X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
error_rates = []
k_range = range(1, 21)

def train_model():
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        pred_k = knn.predict(X_test)
        error_rates.append(1 - accuracy_score(y_test, pred_k))

# Визуализация ошибок
# plt.figure(figsize=(10,6))
# plt.plot(k_range, error_rates, marker='o')
# plt.title('Error Rate vs. K Value')
# plt.xlabel('K')
# plt.ylabel('Error Rate')
# plt.xticks(k_range)
# plt.grid(True)
# plt.show()

def make_prediction():
    optimal_k = 5
    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    knn.fit(X_train, y_train)
    return knn.predict(X_test)

def get_knn_prediction():
    train_model()
    y_pred = make_prediction()
    report = classification_report(y_test, y_pred, output_dict=True)

    return report

# print("Classification Report:\n", classification_report(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
