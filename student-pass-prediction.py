import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = {
    'hours_studied': [1, 2, 3, 4, 5, 6, 7, 8],
    'attendance': [50, 60, 65, 70, 75, 80, 85, 90],
    'pass': [0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)
print(df)

X = df[['hours_studied', 'attendance']]
y = df['pass']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Predictions:", y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

new_data = [[5, 75]]  
result = model.predict(new_data)

if result[0] == 1:
    print("Pass")
else:
    print("Fail")