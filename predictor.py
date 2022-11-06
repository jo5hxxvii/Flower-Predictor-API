from sklearn.linear_model import LogisticRegression as lr
import pickle


with open('flower_model.pkl', 'rb') as file:
        model = pickle.load(file)
labels = ['setosa', 'versicolor', 'virginica']

def predict_flower(df):
    prediction = model.predict(df)
    return labels[prediction[0]]