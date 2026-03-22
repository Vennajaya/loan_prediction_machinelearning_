import pandas as pd

df = pd.read_csv("loan_prediction_dataset.csv")
print(df.head())
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Employment_Status'] = le.fit_transform(df['Employment_Status'])

print(df.head())
from sklearn.model_selection import train_test_split

X = df.drop('Loan_Approved', axis=1)
y = df['Loan_Approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Data split done")
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Model trained successfully")
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix, classification_report

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
import pandas as pd

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

print("\nFeature Importance:")
print(importance.sort_values(by='Importance', ascending=False))
import pickle

pickle.dump(model, open("model.pkl", "wb"))