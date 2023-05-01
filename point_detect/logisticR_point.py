"""
link: https://www.kaggle.com/code/shrutimechlearn/lower-back-pain-data-87-highest-acc-detailed
有图
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import scikitplot as skplt

# over_samples = SMOTE(random_state=0)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('./point_quality2.csv')
# data = pd.read_csv('./point_quality.csv')[:200]
# print(data)
data['label'] = data['label'].astype('category')
encode_map = {
    'bad': 1,
    'good': 0
}

data['label'].replace(encode_map, inplace=True)

sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(data.drop(["Q", "score", "error", "label"], axis=1)))
# X = pd.DataFrame(sc_X.fit_transform(data.drop(["D", "Q", "score", "error", "label"], axis=1)))
# X = pd.DataFrame(sc_X.fit_transform(data.drop(["error", "label"], axis=1)))
y = data.label

# random_state = 1, 70
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=70, stratify=y)
# print(X_train, y_train)

# X_train,y_train = over_samples.fit_resample(X_train, y_train)
# print(over_samples_X, over_samples_y)

test_scores = []
train_scores = []

# print("-------KNN-------")
# op_knn = KNeighborsClassifier(n_neighbors=41)
# op_knn.fit(X_train,y_train)
# knn_pred = op_knn.predict(X_test)
# print(classification_report(y_test, knn_pred))
lr_final = LogisticRegression(C=10)

print("-------逻辑回归-------")
log_rg = LogisticRegression().fit(X_train, y_train)
cross_val_score(log_rg, X_train, y_train, cv=5)
# print(log_rg.score(X_test, y_test))
lr_pred = log_rg.predict(X_test)
lr_acc = accuracy_score(y_test , lr_pred)*100
print(lr_acc)
print(classification_report(y_test, lr_pred, digits=4))
skplt.metrics.plot_roc_curve(y_test, log_rg.predict_proba(X_test), figsize=(6,6))
plt.tight_layout()
plt.show()

print("Auc Score: {}".format(roc_auc_score(y_test, lr_pred)))

# print(data.describe())
print("-------随机森林-------")
rf = RandomForestClassifier(max_depth=8, n_estimators=6, min_samples_leaf=4, min_samples_split=8)
rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)
print(classification_report(y_test, rf_pred))
print("Auc Score: {}".format(roc_auc_score(y_test, rf_pred)))
#
# print("-------决策树-------")
# dt = DecisionTreeClassifier(max_depth=5, max_features=2, min_samples_leaf=7)
# dt.fit(X_train, y_train)
# dt_pred = dt.predict(X_test)
# print(classification_report(y_test, dt_pred))
#
# print("-------融合-------")
# stacked_pred = np.array([lr_pred, rf_pred, dt_pred])
# stacked_pred = np.transpose(stacked_pred)
# lr_final.fit(stacked_pred, y_test)
# final = lr_final.predict(stacked_pred)
# lr_final_acc = accuracy_score(y_test , final)*100
# print(lr_final_acc)
# print(classification_report(y_test,final))

# print("-------SVC-------")
# lr_svc = SVC(kernel = 'linear', gamma = 'auto',probability=True)
# lr_svc.fit(X_train,y_train)
# svc_pred = lr_svc.predict(X_test)
# print(classification_report(y_test, svc_pred))

# skplt.metrics.plot_roc_curve(y_test, lr_final.predict_proba(stacked_pred), figsize=(6,6))
# plt.show()
#
# print("Auc Score: {}".format(roc_auc_score(y_test,final)))