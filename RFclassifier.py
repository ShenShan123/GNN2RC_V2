import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#在two_moon数据集上构造5棵树组成的随机森林
import seaborn as sns
import mglearn
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
 
X,y = make_moons(n_samples=100,noise=0.25,random_state=3)
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=42)
 
RFC = RandomForestClassifier(n_estimators=100,random_state=2).fit(X_train,y_train)
 
# 将每棵树及总预测可视化
fig,axes = plt.subplots(2,3,figsize=(20,10))
for i,(ax,tree) in enumerate(zip(axes.ravel(),RFC.estimators_)):
    ax.set_title('Tree{}'.format(i))
    mglearn.plots.plot_tree_partition(X_train,y_train,tree,ax=ax)
mglearn.plots.plot_2d_separator(RFC,X_train,fill=True,ax=axes[-1,-1],alpha=.4)
axes[-1,-1].set_title('Random Forest')
mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
plt.show()