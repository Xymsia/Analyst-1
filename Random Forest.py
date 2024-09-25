import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

data=pd.read_csv(&#39;heart.csv&#39;)
dataX=data.drop(&#39;target&#39;,axis=1)
dataY=data[&#39;target&#39;]
X_train,X_test,y_train,y_test=train_test_split(dataX,dataY,test_size=0.2,random_state=42)
X_train=(X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train)).values
X_test=(X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test)).values
pca=PCA().fit(X_train)
pca = PCA(n_components=8)
pca.fit(X_train)
pca = PCA(n_components=8)
pca.fit(X_test)
reduced_data_train = pca.transform(X_train)
reduced_data_test = pca.transform(X_test)
reduced_data_train = pd.DataFrame(reduced_data_train,
columns=[&#39;Dim1&#39;,&#39;Dim2&#39;,&#39;Dim3&#39;,&#39;Dim4&#39;,&#39;Dim5&#39;,&#39;Dim6&#39;,&#39;Dim7&#39;,&#39;Dim8&#39;])
reduced_data_test = pd.DataFrame(reduced_data_test,
columns=[&#39;Dim1&#39;,&#39;Dim2&#39;,&#39;Dim3&#39;,&#39;Dim4&#39;,&#39;Dim5&#39;,&#39;Dim6&#39;,&#39;Dim7&#39;,&#39;Dim8&#39;])
X_train=reduced_data_train
X_test=reduced_data_test
def plot_roc_(false_positive_rate,true_positive_rate,roc_auc):
plt.figure(figsize=(5,5))
plt.title(&#39;Receiver Operating Characteristic&#39;)
plt.plot(false_positive_rate,true_positive_rate, color=&#39;red&#39;,label = &#39;AUC = %0.2f&#39; % roc_auc)
plt.legend(loc = &#39;lower right&#39;)
plt.plot([0, 1], [0, 1],linestyle=&#39;--&#39;)
plt.axis(&#39;tight&#39;)
plt.ylabel(&#39;True Positive Rate&#39;)
plt.xlabel(&#39;False Positive Rate&#39;)
plt.show()
def plot_feature_importances(gbm):
n_features = X_train.shape[1]

plt.barh(range(n_features), gbm.feature_importances_, align=&#39;center&#39;)
plt.yticks(np.arange(n_features), X_train.columns)
plt.xlabel(&quot;Feature importance&quot;)
plt.ylabel(&quot;Feature&quot;)
plt.ylim(-1, n_features)
combine_features_list=[
(&#39;Dim1&#39;,&#39;Dim2&#39;,&#39;Dim3&#39;),
(&#39;Dim4&#39;,&#39;Dim5&#39;,&#39;Dim5&#39;,&#39;Dim6&#39;),
(&#39;Dim7&#39;,&#39;Dim8&#39;,&#39;Dim1&#39;),
(&#39;Dim4&#39;,&#39;Dim8&#39;,&#39;Dim5&#39;)
]
#RANDOM FOREST
parameters = [
{
&#39;max_depth&#39;: np.arange(1, 10),
&#39;min_samples_split&#39;: np.arange(2, 5),
&#39;random_state&#39;: [3],
&#39;n_estimators&#39;: np.arange(10, 20)
},
]
for features in combine_features_list:
X_train_set=X_train.loc[:,features]
X_test1_set=X_test.loc[:,features]
tree=GridSearchCV(RandomForestClassifier(),parameters,scoring=&#39;accuracy&#39;)
tree.fit(X_train_set, y_train)
print(&#39;Best parameters set:&#39;)
print(tree.best_params_)
predictions = [
(tree.predict(X_train_set), y_train, &#39;Train&#39;),
(tree.predict(X_test1_set), y_test, &#39;Test1&#39;)
]
rfc=RandomForestClassifier(max_depth=7,min_samples_split=4,n_estimators=19,random_state=3)
rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)
y_proba=rfc.predict_proba(X_test)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_proba[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)
plot_roc_(false_positive_rate,true_positive_rate,roc_auc)
from sklearn.metrics import accuracy_score
print(&#39;Accurancy Oranı :&#39;,accuracy_score(y_test, y_pred))
print(&quot;RandomForestClassifier TRAIN score with &quot;,format(rfc.score(X_train, y_train)))
print(&quot;RandomForestClassifier TEST score with &quot;,format(rfc.score(X_test, y_test)))
print()