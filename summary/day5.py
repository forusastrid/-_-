- 26.py

mglearn.plots.plot_knn_regression(n_neighbors=1)

- 27.py

mglearn.plots.plot_knn_regression(n_neighbors=3)

- 28.py

from sklearn.neighbors import KNeighborsRegressor

X,y : mglearn.datasets.make_wave(n_samples=40)

X_train , X_test , y_train , y_test = train_test_split(X,y,random_states=0)

reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train, y_train)

- 29.py

print("테스트 세트 예측 :\n", reg.predict(X_test))

- 30.py

print("테스트 세트 R^2 : {:2f}".format(reg.score(X_test, y_test)))

- 31.py

fig, axes = plt.subplots(1,3, figsize=(15,4))

line = np.linspace(-3,3,1000).reshape(-1, 1)
for n_neighbors, ax in zip([1,3,9], axes):
  reg = KNeighborsRegressor(n_neighbors=n_neighbors)
  reg.fit(X_train, y_train)
  ax.plot(line,reg.predict(line))
  ax.plot(X_train, y_train, '^', c=mglearn.cm2(0) ,markersize=8)
  ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1) ,markersize=8)

  ax.set_title(
    "{} 이웃의 훈련 스코어: {: 2f} 테스트 스코어: {: 2f}".format(
      n_neighbors, reg.score(X_train, y_train), reg.score(X_test, y_test)))
  ax.set_xlabel("특성")
  ax.set_ylabel("타깃")
axes[0].legend(["모델 예측", "훈련 데이터/타깃" , "테스트 데이터/타깃"], loc="best"
plt.show()
               
- 32.py
from sklearn.linear_model import LinearRegression
X,y : mglearn.datasets.make_wave(n_samples=60)

X_train , X_test , y_train , y_test = train_test_split(X,y,random_states=42)

lr = LinearRegression().fit(X_train,y_train)

print("lr.coef_ :" , lr.coef_)
print("lr.intercept_ :" , lr.intercept_)

print("훈련 세트 점수 : {: 2f}".format(lr.score(X_train, y_train)))
print("테스트 세트 점수 : {: 2f}".format(lr.score(X_test, y_test)))

- 33.py

X,y : mglearn.datasets.load_extended_boston()

X_train , X_test , y_train , y_test = train_test_split(X,y,random_states=0)
lr = LinearRegression().fit(X_train,y_train)

print("훈련 세트 점수 : {: 2f}".format(lr.score(X_train, y_train)))
print("테스트 세트 점수 : {: 2f}".format(lr.score(X_test, y_test)))
