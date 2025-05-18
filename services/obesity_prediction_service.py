import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import io
import seaborn as sns
from sklearn.metrics import confusion_matrix
from schemas.dataset_schema import ObesityInput
from schemas.user_input_schema import UserInput
import numpy as np
df = None
label_encoders = {}
target_encoder = None
scaler = None
X_train, X_test, y_train, y_test = None, None, None, None
categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC',
                    'SMOKE', 'SCC', 'CALC', 'MTRANS']
numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
error_rates = []
k_range = range(1, 21)


def load_data_from_excel():
    return pd.read_excel("datasets/obes_dataset.xlsx")


def initialize():
    global df, label_encoders, target_encoder, scaler, X_train, X_test, y_train, y_test

    df = load_data_from_excel()

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(df[col])
        label_encoders[col] = le

    target_encoder = LabelEncoder()
    target_encoder.fit(df['NObeyesdad'])

    scaler = StandardScaler()
    scaler.fit(df[numerical_cols])

    df_encoded = df.copy()
    for col in categorical_cols:
        df_encoded[col] = label_encoders[col].transform(df_encoded[col])
    df_encoded[numerical_cols] = scaler.transform(df_encoded[numerical_cols])
    df_encoded['NObeyesdad'] = target_encoder.transform(df_encoded['NObeyesdad'])

    X = df_encoded.drop('NObeyesdad', axis=1)
    y = df_encoded['NObeyesdad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def train_model():
    error_rates.clear()
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        pred_k = knn.predict(X_test)
        error_rates.append(1 - accuracy_score(y_test, pred_k))


def get_knn_error_plot():
    train_model()
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, error_rates, marker='o')
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf.read()


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


def add_new_data_and_retrain(data: ObesityInput):
    global df
    df_existing = load_data_from_excel()
    new_row = pd.DataFrame([data.model_dump()])
    df_new = pd.concat([df_existing, new_row], ignore_index=True)
    df_new.to_excel("datasets/obes_dataset.xlsx", index=False)

    initialize()


def preprocess_user_input(user_data: UserInput):
    df_user = pd.DataFrame([user_data.model_dump()])

    for col in categorical_cols:
        le = label_encoders[col]
        df_user[col] = le.transform(df_user[col])

    df_user[numerical_cols] = scaler.transform(df_user[numerical_cols])
    return df_user


def predict_obesity_class(user_df):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    pred_encoded = knn.predict(user_df)
    pred_class = target_encoder.inverse_transform(pred_encoded)
    return pred_class[0]


def plot_confusion_matrix():
    train_model()
    y_pred = make_prediction()
    cm = confusion_matrix(y_test, y_pred)
    labels = target_encoder.inverse_transform(np.unique(y_test))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

initialize()
