# Generalised Program For Classifiers:

<aside>
🤖 **Classifiers Includes :**

1. Logistic Regression
2. Decison Tree
3. K Nearest Neighbors
4. Naive Baiyes
5. Support Vector 
6. Ensemble Techniques
    1. Bagging
    2. Boosting
    3. Random Forest
</aside>

<aside>
💀 **A Generalised Program For Classifiers :**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selectionimport train_test_split
from sklearn.preprocessingimport StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score
from sklearn import tree
```

```python
dataset = pd.read_csv('Logistic_Iris.csv')
x = dataset.iloc[:, [0,1,2,3]].values
y = dataset.iloc[:, 4].values
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=0)
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)
```

```python
classifier = LogisticRegression(random_state = 0)
classifier = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=3, min_samples_leaf=5)
classifier = KNeighborsClassifier(n_neighbors=7)
classifier = GaussianNB()
classifier = SVC(kernel = 'linear', random_state = 0)
classifier = BaggingClassifier(base_estimator = GaussianNB(), n_estimators = 100, random_state= 0)
classifier = AdaBoostClassifier(n_estimators = 50, learning_rate = 0.2)
classifier = RandomForestClassifier(n_estimators = 100)

```

```python
classifier.fit(xtrain, ytrain)
y_pred = classifier.predict(xtest)
```

```python
acc = accuracy_score(ytest, y_pred)*100
cm =confusion_matrix(ytest, y_pred)
# some time you may also include :
pres=precision_score(ytest, y_pred,average='macro')
tpr = recall_score(ytest, y_pred,average='macro')
fpr = 1 - tpr
```

```python
# General Code for plotting Confusion Matrix

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",cbar=False,xticklabels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],yticklabels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
plt.title("Confusion Matrix for Logistic Regression Classifier")
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.show()**⬇️ Only for Decision Tree :**
```

</aside>

<aside>
💀 **Logistic Regression :**

- Note that you can change the values of x and y with :
    - `Petal Length`
    - `Peatal Width`
    - `Sepal Length'`
    - `Sepal Width'`
    
    ```python
    sns.regplot(x='Sepal Length',y='Sepal Width',data=dataset)
    plt.show()
    ```
    
</aside>

<aside>
💀 Confusion Matrix for All Classifiers :

```python
# Create a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",cbar=False,xticklabels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],yticklabels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
plt.title("Confusion Matrix for Classifier")
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.show()
```

</aside>

<aside>
💀 Decision Tree :

```python
tree.plot_tree(dtree_gini , rounded = True,fontsize = 10)
```

</aside>

<aside>
💀 Performace Matrix :

- Just change the values of y with : [Acc,Pres,Tpr,Fpr] one after the other

```python
plt.subplots(figsize=(3,3))
labels=['LoR', 'NB']
sns.barplot(x=labels,y=Tpr,palette='hot',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Classifier Comparison based on True Possitive Rate')
plt.show()
```

</aside>