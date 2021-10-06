import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def create_data(N = 1000):
    X_men = np.random.randn(N//2) * 4 + 70
    X_women = np.random.randn(N//2) * 3.5 + 65

    return X_men, X_women

def prob():
    x = int(input("Enter test height:"))
    p_x_women_num = norm.pdf(x, loc = 65, scale = 3.5) * 0.5 #Calculates the numerator of Bayes
    p_x_men_num = norm.pdf(x, loc = 70, scale = 4) * 0.5     #Calculates the numerator of Bayes

    p_x = p_x_men_num + p_x_women_num #Calculates the denominator

    p_x_men = p_x_men_num/p_x
    p_x_women = p_x_women_num/p_x

    if p_x_women > p_x_men:
        print("Predict women: ", p_x_women)
    elif p_x_men == p_x_women:
        print("Equal chance to be male or female")
    else:
        print("Predict men: ", p_x_men)

if __name__ == "__main__":
    #X_men, X_women, num = create_data()
    prob()

    