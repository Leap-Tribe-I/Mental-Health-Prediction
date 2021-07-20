#importing basic python libraries
import pandas as pd

#importing ploting libraries
import matplotlib.pyplot as plt
import seaborn as sns

#creating a covarinance matrix of the encoded data to visualize correlation between data points
def corrMat(data):
    corr = data.corr()
    #printing the Covarinance matrix
    print("\n")
    print("Correlation Matrix:\n")
    print(corr)
    print("\n")
    f, ax = plt.subplots(figsize=(9, 9))
    sns.heatmap(corr, vmax=.8, square=True, annot=True)
    plt.show()