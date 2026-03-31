import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

data = pd.read_csv("data/student_data.csv")

X = data.drop("final_result", axis=1)
y = data["final_result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

pickle.dump(model, open("model/model.pkl", "wb"))
