#importing basic python libraries
import pandas as pd
import numpy as np

#importing ploting libraries
import matplotlib.pyplot as plt

#ploting the accuracy graph of all the models 
def accuracy_graph(accuracyDict):
    s = pd.Series(accuracyDict)
    s = s.sort_values(ascending=False)
    plt.figure(figsize=(12,8))
    ax = s.plot(kind='bar') 
    for p in ax.patches:
        ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.ylim([70.0, 90.0])
    plt.xlabel('Method')
    plt.ylabel('Percentage')
    plt.title('Success of methods')
    plt.show()
    plt.savefig('Accuracy Graph.png')