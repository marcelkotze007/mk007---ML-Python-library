import numpy as np
from scipy.stats import norm

def create_data():
    X = np.array([
        [1, 1, 1, 1],
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
        [1, 1, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    Y = X[:, -1]
    X = X[:, :-1]

    return X, Y

def prob():
    #X, Y = create_data()

    #X_money = X[:,0]
    #X_free = X[:, 1]
    #X_pills = X[:, 2]

    #Must first calculate each probality for a given outcome
    #First do p(money):
    #p(X_money | Y = 1)
    p_mon_1 = 3/5
    p_not_mon_1 = 1 - p_mon_1
    #p(X_money | Y = 0)
    p_mon_0 = 2/5
    p_not_mon_0 = 1 - p_mon_0

    #Next do p(free)
    #p(X_free | Y = 1)
    p_fr_1 = 4/5
    p_not_fr_1 = 1 - p_fr_1 
    #p(X_free | Y = 0)
    p_fr_0 = 1/5
    p_not_fr_0 = 1 - p_fr_0
    
    #Lastly do p(pills)
    #p(X_pills | Y = 1)
    p_pil_1 = 4/5
    p_not_pil_1 = 1 - 4/5
    #p(X_pills | Y = 0)
    p_pil_0 = 0
    p_not_pil_0 = 1 - p_pil_0

    return (p_fr_0, p_fr_1, p_not_fr_0, p_not_fr_1, p_mon_0, p_mon_1, p_not_mon_0, p_not_mon_1,
            p_pil_0, p_pil_1, p_not_pil_0, p_not_pil_1)

def calc_prob():
    p_fr_0, p_fr_1, p_not_fr_0, p_not_fr_1, p_mon_0, p_mon_1, p_not_mon_0, p_not_mon_1, p_pil_0, p_pil_1, p_not_pil_0, p_not_pil_1 = prob()
    #Calculate the p(spam | money, ~free, ~pills) = P(X | Y) * P(Y)
    p_spam = p_mon_1 * p_not_fr_1 * p_not_pil_1 * 0.5
    #Not true prob, only use to determine to see if it is spam or not
    #Calculate the p(~spam | money, ~free, ~pills) = p(X | Y) * P(Y)
    p_non_spam = p_mon_0 * p_not_fr_0 * p_not_pil_0 * 0.5
    
    if p_non_spam > p_spam:
        print('Not spam')

if __name__ == "__main__":
    calc_prob()