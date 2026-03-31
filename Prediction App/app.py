import pickle
import numpy as np

model = pickle.load(open("model/model.pkl", "rb"))

def predict(hours, attendance, sleep, prev_score):
    data = np.array([[hours, attendance, sleep, prev_score]])
    return model.predict(data)

print(predict(5, 80, 7, 60))
