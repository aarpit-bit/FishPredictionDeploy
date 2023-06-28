import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from hyperopt import hp
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from sklearn.model_selection import cross_val_score
data = pd.read_csv("Fish.csv")

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

data_cleaned = data.drop("Weight", axis=1)
y = data['Weight']
x_train, x_test, y_train, y_test = train_test_split(data_cleaned, y, test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# label encoder
label_encoder = LabelEncoder()
x_train['Species'] = label_encoder.fit_transform(x_train['Species'].values)
x_test['Species'] = label_encoder.transform(x_test['Species'].values)


def evauation_model(pred, y_val):
    score_MSE = round(mean_squared_error(pred, y_val),2)
    score_MAE = round(mean_absolute_error(pred, y_val),2)
    score_r2score = round(r2_score(pred, y_val),2)
    return score_MSE, score_MAE, score_r2score


def models_score(model_name, train_data, y_train, val_data,y_val):

    model_list = ["Decision_Tree", "Random_Forest", "XGboost_Regressor"]
    # model_1
    if model_name == "Decision_Tree":
        reg = DecisionTreeRegressor(random_state=42)
    # model_2
    elif model_name == "Random_Forest":
        reg = RandomForestRegressor(random_state=42)

    # model_3
    elif model_name == "XGboost_Regressor":
        reg = xgb.XGBRegressor(objective="reg:squarederror",random_state=42,)
    else:
        print("please enter correct regressor name")

    if model_name in model_list:
        reg.fit(train_data, y_train)
        pred = reg.predict(val_data)

        score_MSE, score_MAE, score_r2score = evauation_model(pred, y_val)
        return round(score_MSE, 2), round(score_MAE, 2), round(score_r2score, 2)


model_list = ["Decision_Tree", "Random_Forest", "XGboost_Regressor"]
result_scores = []
for model in model_list:
    score = models_score(model, x_train, y_train, x_test, y_test)
    result_scores.append((model, score[0], score[1], score[2]))
    print(model, score)

df_result_scores = pd.DataFrame(result_scores)
print(df_result_scores)

num_estimator = [100,150,200,250]

space = {'max_depth': hp.quniform("max_depth", 3, 18, 1),
         'gamma': hp.uniform ('gamma', 1,9),
         'reg_alpha' : hp.quniform('reg_alpha', 30,180,1),
         'reg_lambda' : hp.uniform('reg_lambda', 0,1),
         'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
         'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
         'n_estimators': hp.choice("n_estimators", num_estimator),
         }


def hyperparameter_tuning(space):
    model = xgb.XGBRegressor(n_estimators=space['n_estimators'], max_depth = int(space['max_depth']), gamma=space['gamma'],
                             reg_alpha=int(space['reg_alpha']), min_child_weight=space['min_child_weight'],
                             colsample_bytree=space['colsample_bytree'], objective="reg:squarederror")

    score_cv = cross_val_score(model, x_train, y_train, cv=5, scoring="neg_mean_absolute_error").mean()
    return {'loss':-score_cv, 'status': STATUS_OK, 'model': model}


trials = Trials()
best = fmin(fn=hyperparameter_tuning,
            space=space,
            algo=tpe.suggest,
            max_evals=200,
            trials=trials)

print("This is best: ", best)
best['max_depth'] = int(best['max_depth'])  # convert to int
best["n_estimators"] = num_estimator[best["n_estimators"]]  #assing value based on index
reg = xgb.XGBRegressor(**best)
reg.fit(x_train, y_train)
pred = reg.predict(x_test)
score_MSE, score_MAE, score_r2score = evauation_model(pred, y_test)
to_append = ["XGboost_hyper_tuned", score_MSE, score_MAE, score_r2score]
df_result_scores.loc[len(df_result_scores)] = to_append
df_result_scores

# winner
reg = xgb.XGBRegressor(**best)
reg.fit(x_train, y_train)
pred = reg.predict(x_test)
plt.figure(figsize=(18, 7))
plt.subplot(1, 2, 1)  # row 1, col 2 index 1
plt.scatter(range(0, len(x_test)), pred, color="green",label="predicted")
plt.scatter(range(0, len(x_test)), y_test, color="red",label="True value")
plt.legend()

plt.subplot(1, 2, 2)  # index 2
plt.plot(range(0, len(x_test)), pred, color="green", label="predicted")
plt.plot(range(0, len(x_test)), y_test, color="red", label="True value")
plt.legend()
plt.show()


