
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

model = joblib.load('model.pkl')

test_df = pd.read_csv("iris.csv")
test_df = test_df.drop("Id", axis=1)

X = test_df.drop("Species", axis=1)
y = test_df["Species"]

# Handle missing values
X = X.fillna(X.median(numeric_only=True))
X = X.fillna(0)

pred = model.predict(X)
accuracy = accuracy_score(y, pred)

print(accuracy)

assert accuracy >= 0.98
assert len(pred) == len(y)

print("All tests passed successfully!")
