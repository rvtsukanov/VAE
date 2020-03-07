import pandas as pd
from vae import VAE
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost


regressor = xgboost.XGBRegressor()
regressor.load_model('./xgb_reg')

train_set = pd.read_csv("./X_train.csv")
y = pd.read_csv('./Y_train.csv')['Температура']

print(f'max y: {y.max()} min y: {y.min()}')

model = VAE(1200, dim_z=8, train_set=train_set).load_model("last")

# model.run().save_model().save_loss()

feature_no = 12
object_no = 1000


num_sampled_points = 10000
tr, _ = model.generate_from_noise(train_set.iloc[object_no : object_no + 1], num_sampled_points)

# TODO: check shapes! some shit accured

generated_targets = regressor.predict(tr[:, 0, :])


sns.set()
sns.distplot(generated_targets)
# sns.distplot(y)
plt.title(f'Targets from {y[object_no]}')

plt.scatter(
        y[object_no],
        [0],
        c="r",
        label=f"original point = {y[object_no]:.1f}",
    )

plt.scatter(y.mean(), [0], c='b', label=f'mean_target = {y.mean():.1f}')
plt.scatter(y.median(), [0], c='g', label=f'median_target = {y.median():1f}')
plt.legend()

plt.show()




print(tr.shape)
tr_np = tr[:, 0, feature_no]


def render_one_feature_plot(object_no, feature_no):
    sns.set()
    sns.distplot(tr_np)

    plt.scatter(
        train_set.iloc[object_no, feature_no],
        [0],
        c="r",
        label=f"original point = {train_set.iloc[object_no, feature_no]:.3f}",
    )
    plt.scatter(
        train_set.iloc[:, feature_no].mean(), [0], c="g", label=f"mean = {train_set.iloc[:, feature_no].mean():.3f}",
    )
    plt.scatter(
        train_set.iloc[:, feature_no].median(),
        [0],
        c="b",
        label=f"median = {train_set.iloc[:, feature_no].median():.3f}",
    )
    plt.legend()

    plt.title(train_set.columns[feature_no] + f" {num_sampled_points} points")
    # plt.show()


render_one_feature_plot(object_no, feature_no)
