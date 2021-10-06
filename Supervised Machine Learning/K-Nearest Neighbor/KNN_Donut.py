import numpy as np
import matplotlib.pyplot as plt

from Utilities_KNN import KNN_Util
from KNN import KNN

if __name__ == "__main__":
    KNN_Util = KNN_Util()
    X, Y = KNN_Util.get_donut(N=200)

    k = int(input("Enter the k value:"))

    model = KNN(k)
    model.fit(X, Y)
    print("Accuracy:", model.score(X, Y))

    plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
    plt.title("Donut Problem")
    plt.show()


