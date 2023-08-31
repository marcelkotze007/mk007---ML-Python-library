# import tensorflow as tf 
# print(tf.__version__)

import numpy as np 
from time import time

list_data = [4,6,4,84,231,321,6514,132,154,9,21,5,484,41545412,1,54,54,2,45,42,15,451,2,12,-5,-5,-64,-454,-6,-9,-56532,-100]

# Test loop speed first
start = time()
highest_value = 0
for i in list_data: 
    if i > highest_value and i*-1 in list_data:
        highest_value = i 
end = time()

diff = start - end

print(f"Biggest value that is both possitive and negative is {highest_value} and it took {diff}")