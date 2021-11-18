import pandas as pd
import numpy as np 
from datetime import datetime as dt

from K_Means import K_means

def get_data(limit = 1000):
    data = pd.read_csv(r"C:\Dev\SourceCode\Personal\mk007---ML-Python-library\data\train.csv")
    #data.header()
    dataframe = data.values

    np.random.shuffle(dataframe)
    Y = dataframe[:limit,0]
    X = dataframe[:limit,1:]

    return X, Y

if __name__ == "__main__":
    
    X, Y = get_data(limit=1000)
    
    model = K_means()
    dt0 = dt.now()
    model.fit(X, K=10, max_iter=20,beta=0.2,show_cost=True, show_grid=False, validate="Purity", Y=Y)
    print("Time", dt.now() - dt0)
