import matplotlib.pyplot as plt
import pandas as pd

accuracyDict={
 "Log_Reg_mod_tuning": 85.71428571428571,
 "Log_Reg_mod_tuning_rand": 85.71428571428571,
 "KNN": 90.47619047619048,
 "KNN_rand": 80.95238095238095,
 "Decision_Tree": 80.95238095238095,
 "Decision_Tree_rand": 80.95238095238095,
 "Random_Forest": 85.71428571428571,
 "Random_Forest_rand": 80.95238095238095,
 "AdaBoost": 80.95238095238095,
 "AdaBoost_rand": 80.95238095238095
}

# plot accuracy bar graph
# plt.bar(range(len(accuracyDict)), accuracyDict.values(), align='center')
# plt.xticks(range(len(accuracyDict)), accuracyDict.keys(), rotation=90)
# plt.xlabel('Algorithms')
# plt.ylabel('Accuracy')
# plt.title('Accuracy Graph')
# plt.show()
# plt.savefig('Accuracy Graph.png')

s = pd.Series(accuracyDict)
s = s.sort_values(ascending=False)
plt.figure(figsize=(12,8))
#Colors
ax = s.plot(kind='bar') 
for p in ax.patches:
    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.005, p.get_height() * 1.005))
plt.ylim([70.0, 90.0])
plt.xlabel('Method')
plt.ylabel('Percentage')
plt.title('Success of methods')
plt.show()
plt.savefig('Accuracy Graph.png')