#This forms a KNN model that predicts car classes
import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
#print(data.head())

pre = preprocessing.LabelEncoder()
buying = pre.fit_transform(list(data['buying']))
maint = pre.fit_transform(list(data['maint']))
doors = pre.fit_transform(list(data['doors']))
persons = pre.fit_transform(list(data['persons']))
lug_boot = pre.fit_transform(list(data["lug_boot"]))
safety = pre.fit_transform(list(data['safety']))
cls = pre.fit_transform(list(data['class']))


predict = 'class'
X = list(zip(buying, maint, doors, persons, lug_boot, safety))
y = list(cls)

#Testing best accuracy
best = 0
bestK = 0
accSum = 0
for i in range(5, 11):
    for j in range(100):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(x_train,y_train)
        acc = model.score(x_test,y_test)
        accSum += acc
    averageAcc = accSum / 100
    print(str(i) + ": " + str(averageAcc))
    accSum = 0
    if averageAcc > best:
        bestK = i
        best = averageAcc

print("Highest) " + str(bestK) + ": " + str(best))

names = ["unacc", "acc", "good", "vgood"]

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
model = KNeighborsClassifier(n_neighbors=bestK)
model.fit(x_train,y_train)
predicted = model.predict(x_test)
for x in range(len(predicted)):
    print("Predicted ", names[predicted[x]], "Data ", x_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9)
    print('N: ', n)