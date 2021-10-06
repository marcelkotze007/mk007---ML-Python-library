import numpy as np
import matplotlib.pyplot as plt

#The cost function is J = w^2
#Thus the derivative of dJ/dw = 2w
"""
w = 20
X = []
Y = []

for i in range(100):
    w = w - 0.1*2*w
    print(w)
    Y.append(w)
    X.append(i)

plt.plot(X, Y)
plt.show()
"""

#Exercise:
# J(w1, w2) = w1^2 + w2^4
# dJ/dw1 = 2w
# dJ/dw2 = 4w^3
#Insert into a matrix:
#np.array[w-0.1*2w, w - 0.1*4w^^3]

