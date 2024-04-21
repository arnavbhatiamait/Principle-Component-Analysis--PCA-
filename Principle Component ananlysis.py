# %% [markdown]
# Principle Component Analysis (PCA)

# %% [markdown]
# Importing The Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %% [markdown]
# importing The Data Set

# %%
df=pd.read_csv("Wine.csv")
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
y

# %% [markdown]
# Spliting Data set Into test and training sets

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# %%
x_train

# %%
x_test


# %%
y_train

# %%
y_test

# %% [markdown]
# Feature Scaling

# %%
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# %%
x_test

# %%
x_train

# %% [markdown]
# Applying The PCA

# %%
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)

# %% [markdown]
# Training The Logistic Regression model On Training Set

# %%
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

# %% [markdown]
# Making The Confusion Matrix 

# %%
from sklearn.metrics import confusion_matrix,accuracy_score
y_pred=classifier.predict(x_test)
print(y_pred)
cnf=confusion_matrix(y_test,y_pred)
print(cnf)
accuracy_score(y_test,y_pred)

# %% [markdown]
# Heatmap

# %%
import seaborn as sns
className=[0,1]
fig,ax=plt.subplots()
tick_marks=np.arange(len(className))
plt.xticks(tick_marks,className)
plt.yticks(tick_marks,className)
# heatmap
sns.heatmap(pd.DataFrame(cnf),annot=True,cmap="YlGnBu",fmt="g")
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title("Confusion Matrix",y=1.1)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()
plt.savefig("Confussion Matrix.png")


# %% [markdown]
# 

# %% [markdown]
# Visulaizing the training set result

# %%
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.savefig('Logistic Regression (Training set).png')
plt.show()

# %% [markdown]
# Visulaizing the test set result

# %%
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.savefig('Logistic Regression (Test set).png')
plt.show()

# %% [markdown]
# Visualizing The Training Set Results

# %% [markdown]
# 


