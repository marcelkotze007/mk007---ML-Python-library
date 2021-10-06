import numpy as np
import pandas as pd
from scipy import stats

class t_test(object):
    def __init__(self):
        pass

    def get_data(self):
        data = pd.read_csv('advertisement_clicks.csv')
        a = data[data['advertisement_id'] == A]
        b = data[data['advertisement_id'] == B]
        
        a = a['action']
        b = b['action']

        print('a.mean; ', a.mean())
        print('b.mean; ', b.mean())

        #Test to see if N is the same for both groups
        #count_A = 0
        #count_B = 0
        #for i in range(len(add)):
        #    if "A" == add[i]:
        #        count_A = count_A + 1
        #    else:
        #        count_B = count_B + 1

        #print("A: ", count_A, "B: ", count_B )

    def scipy_t_test(self, a, b):
        t, p = stats.ttest_ind(a, b)
        print('t: ', t, 'p: ', p)
    
    def scipy_welch(self, a, b):
        t, p = stats.ttest_ind(a, b, equal_var=False):
        print('welch-t: ', t, 'welch-p: ', p)

if __name__ == "__main__":
    model = t_test()
    model.get_data()    
