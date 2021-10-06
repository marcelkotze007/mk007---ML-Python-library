import numpy as np 
from scipy.stats import chi2
import matplotlib.pyplot as plt

class data_generator(object):
    def __init__(self, p1, p2):
        self.p1 = p1 #prob of click for group 1
        self.p2 = p2 #prob of click for group 2

    def next(self):
        """
        Will return wheter a person clicked on advertisement 1 or advertisement 2. Also ensures there are the same
        number of samples in each group, even though not needed for chisquare.
        """
        click1 = 1 if (np.random.random() < self.p1) else 0 
        click2 = 1 if (np.random.random() < self.p2) else 0 

        return click1, click2


def get_p_value(T):
    """
    Takes in a contingency table. det = first value * last value - second value * third value
    """
    det = T[0,0] * T[1,1] - T[0,1] * T[1,0]
    c2 = float(det)/T[0].sum() * det/T[1].sum() * T.sum()/T[:,0].sum()/T[:,1].sum()
    p = 1 - chi2.cdf(x=c2, df=1)
    
    return p

def experiment(p1, p2, N):
    data = data_generator(p1, p2)
    p_values = np.empty(N)
    T = np.zeros((2,2)).astype(np.float32) #So we can use it with SKlearn if need be
    for i in range(N):
        c1, c2 = data.next()
        T[0,c1] += 1 #puts clicks in the first column, no-clicks in the second column
        T[1,c2] += 1
        if i < 10: # divide by row and column, if any of them are 0, the formula will break 
            p_values[i] = None
        else:
            p_values[i] = get_p_value(T)
    plt.plot(p_values)
    plt.plot(np.ones(N) * 0.05)
    plt.show()

if __name__ == "__main__":
    experiment(0.1, 0.1, 20000)
