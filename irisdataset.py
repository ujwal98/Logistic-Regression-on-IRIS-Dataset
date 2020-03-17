
# Importing the necessary packages and models
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Importing the data
irisData = pd.read_csv('IRIS.csv')

#To just view data, incase our date requires any preprocessing. The IRIS data is pretty clean, so it won't be requiring any pre-processing
irisData.head()

features = ['sepal_length','sepal_width','petal_length','petal_width']
target = ['species']

# Defining our inputs and outputs
iris_X = irisData[features]
iris_y = irisData[target]

#splitting our data into train_batch and test_batch. Test_size=0.33 means 33% of our data will be split into test_batch. Random_state of '1' ensures that our model returns the same result everytime it is run
X_train,X_test,y_train,y_test = train_test_split(iris_X,iris_y,test_size=0.33,random_state = 1)


# Calling our model, fitting the data and prediciting our test_batch
regressor = LogisticRegression()
regressor.fit(X_train,y_train)
preds = regressor.predict(X_test)


#Evaluating our model
print(accuracy_score(y_test,preds))



