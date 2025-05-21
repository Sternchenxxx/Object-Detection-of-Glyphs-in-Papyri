import pickle

with open(r'D:\FAU\project\results\features\hog_features.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)