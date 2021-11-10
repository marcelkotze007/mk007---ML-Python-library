import pickle
from sklearn import tree

#For storing the trained model and just calling at a later state

model = tree.DecisionTreeClassifier()
model.fit(X, Y)

with open("mymodel.pkl", 'wb') as f:
    pickle.dump(model, f)

# Later
with open ("mymodel.pkl", "rb") as f:
    model = pickle.load(f) #already trained

