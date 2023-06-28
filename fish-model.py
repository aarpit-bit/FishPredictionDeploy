import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import  OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from scipy.stats import loguniform
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV


data = pd.read_csv("Fish.csv")

data_num = data.drop(columns=["Species"])
# fig, axes = plt.subplots(len(data_num.columns)//3, 3, figsize=(15, 6))
# i = 0
# for triaxis in axes:
#     for axis in triaxis:
#         data_num.hist(column = data_num.columns[i], ax=axis)
#         i = i+1
# plt.show()

sns.displot(
    data=data,
    x="Weight",
    hue="Species",
    kind="hist",
    height=6,
    aspect=1.4,
    bins=15
)
plt.show()
sns.pairplot(data, kind='scatter', hue='Species');

plt.figure(figsize=(7,6))
corr = data_num.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=True)
plt.show()

ct = make_column_transformer(
    (StandardScaler(),['Length1','Length2','Length3','Height','Width']), #turn all values from 0 to 1
    (OneHotEncoder(handle_unknown="ignore"), ["Species"])
)
# create X and y values
data_cleaned = data.drop("Weight",axis=1)
y = data['Weight']

x_train, x_test, y_train, y_test = train_test_split(data_cleaned,y, test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
X_train_normal = pd.DataFrame(ct.fit_transform(x_train))
X_test_normal = pd.DataFrame(ct.transform(x_test))


def models_score(model_name, train_data, y_train, val_data,y_val):
    model_list = ["Linear_Regression","Lasso_Regression","Ridge_Regression"]
    # model_1
    if model_name=="Linear_Regression":
        reg = LinearRegression()
    # model_2
    elif model_name=="Lasso_Regression":
        reg = Lasso(alpha=0.1,tol=0.03)

    # model_3
    elif model_name=="Ridge_Regression":
        reg = Ridge(alpha=1.0)
    else:
        print("please enter correct regressor name")

    if model_name in model_list:
        reg.fit(train_data,y_train)
        pred = reg.predict(val_data)

        score_MSE = mean_squared_error(pred, y_val)
        score_MAE = mean_absolute_error(pred, y_val)
        score_r2score = r2_score(pred, y_val)
        return round(score_MSE, 2), round(score_MAE, 2), round(score_r2score, 2)


model_list = ["Linear_Regression","Lasso_Regression","Ridge_Regression"]
result_scores = []
for model in model_list:
    score = models_score(model,X_train_normal,y_train, X_test_normal,y_test)
    result_scores.append((model, score[0], score[1],score[2]))
    print(model, score)
df_result_scores = pd.DataFrame(result_scores,columns=["model","mse","mae","r2score"])
print(df_result_scores)

space = dict()
space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']
space['alpha'] = loguniform(1e-5, 50)
model = Ridge()
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
search = RandomizedSearchCV(model, space, n_iter=100,
                            scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv,                 random_state=42)
result = search.fit(X_train_normal, y_train)
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

reg = Ridge(alpha=0.24171039031894245, solver ="sag" )
reg.fit(X_train_normal,y_train)
pred = reg.predict(X_test_normal)
score_MSE = mean_squared_error(pred, y_test)
score_MAE = mean_absolute_error(pred, y_test)
score_r2score = r2_score(pred, y_test)
to_append = ["Ridge_hyper_tuned",round(score_MSE,2), round(score_MAE,2), round(score_r2score,2)]
df_result_scores.loc[len(df_result_scores)] = to_append
print(df_result_scores)

reg = LinearRegression()
reg.fit(X_train_normal, y_train)
pred = reg.predict(X_test_normal)
plt.figure(figsize=(18, 7))
plt.subplot(1, 2, 1)  # row 1, col 2 index 1
plt.scatter(range(0, len(X_test_normal)), pred, color="green", label="predicted")
plt.scatter(range(0, len(X_test_normal)), y_test, color="red", label="True value")
plt.legend()

plt.subplot(1, 2, 2)  # index 2
plt.plot(range(0, len(X_test_normal)), pred, color="green",label="predicted")
plt.plot(range(0, len(X_test_normal)), y_test, color="red",label="True value")
plt.legend()
plt.show()

