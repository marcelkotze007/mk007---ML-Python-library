import numpy as np
from scipy import stats

def create_data(N = 10):
    a = np.random.randn(N) + 2 #mean = 2, var = 1
    b = np.random.randn(N)  #mean = 0, var = 1

    return a, b

def variance(a,b):
    var_a = a.var(ddof = 1) #Numpy uses the population (N) and not sample (N-1), thus pass ddof = 1
    var_b = b.var(ddof = 1)

    return var_a, var_b

def pooledstd(var_a, var_b):
    """
    Calculates the pooled standard deviation (Sp)
    """
    sp = np.sqrt((var_a + var_b) / 2)
    
    return sp

def t_stat(a, b, sp, N = 10):
    """
    Calculates the t-statistic and the degrees of freedom (df)
    """
    t = (a.mean() - b.mean())/(sp * np.sqrt(2/N))

    df = 2*N - 2

    return t, df

def p_value():
    N = 10
    a, b = create_data(N = N)
    var_a, var_b = variance(a, b)
    sp = pooledstd(var_a, var_b)
    t, df = t_stat(a, b, sp, N = N)
    p = 1 - stats.t.cdf(t, df)
    p = p*2 #Since it is a 2-tailed test
    print('t-stat: ', t,'P: ', p)
    t2, p2 = stats.ttest_ind(a, b)
    print('t-stat scipy: ', t2,'P scipy: ', p2)


if __name__ == "__main__":
    p_value()

