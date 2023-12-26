from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
import pickle


data = pd.read_csv('./point_quality.csv')

data['label'] = data['label'].astype('category')
encode_map = {
    'bad': 1,
    'good': 0
}

data['label'].replace(encode_map, inplace=True)

sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(data.drop(["Q", "score", "error", "label"], axis=1)))
y = data.label

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=70, stratify=y)

test_scores = []
train_scores = []

lr_final = LogisticRegression(C=10)

log_rg = LogisticRegression().fit(X_train, y_train)
cross_val_score(log_rg, X_train, y_train, cv=5)
# print(log_rg.score(X_test, y_test))
lr_pred = log_rg.predict(X_test)
lr_acc = accuracy_score(y_test , lr_pred)*100
print(lr_acc)
print(classification_report(y_test, lr_pred, digits=4))

# save model
model_name = "./quality_predict.pkl"
with open(model_name, 'wb') as file:
    pickle.dump(log_rg, file)

skplt.metrics.plot_roc_curve(y_test, log_rg.predict_proba(X_test), figsize=(6,6))
plt.tight_layout()
plt.show()

print("Auc Score: {}".format(roc_auc_score(y_test, lr_pred)))