#/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('results.txt', sep=' ')

# Total recall
recall=df['good'].sum()/df['tot'].sum()
print('recall: ',recall)

# Per class recall
df['recall']=df['good']/df['tot']
weighted_rec=df['recall'].mean()
print('Weighted recall: ',weighted_rec)



ax=df.plot.bar(x='class',y='recall',rot=45, width=.9, edgecolor='black',label='Per class')
plt.plot(np.arange(len(df)),recall*np.ones(len(df)),'--', linewidth=0.5,color='black', label='Average {}'.format(np.round(recall,decimals=2)))
plt.plot(np.arange(len(df)),weighted_rec*np.ones(len(df)),'-.', linewidth=0.5,color='red', label='Weighted Average {}'.format(np.round(weighted_rec,decimals=2)))
ax.set_ylabel('Recall')
ax.tick_params(axis='x', which='major', labelsize=5)
ax.legend(loc='lower right')

for i in np.arange(len(df)):
    plt.annotate(str(df['good'][i])+'/'+str(df['tot'][i]), xy=(i, 0.7), fontsize=5)

plt.show()
