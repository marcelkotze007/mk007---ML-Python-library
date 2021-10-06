import numpy as np

#K = 5,8,10 
#N = length
scores = []
size = N/K
for i in range(K):
        #selects rows between k*size and (k+1)*size
        X_validate, Y_validate = X[i * size: (i+1) * size], Y[i * size: (i+1) * size] 
        #Selects all the values not in validate
        X_train = np.concatenate(X[0 : i * size], X[(i+1) * size: N])
        Y_train = np.concatenate(Y[:i*size], Y[(i+1) * size:])

        model.fit(X_train, Y_train)
        scores.append(model.score(X_validate, Y_validate))
return scores
