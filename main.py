import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

# Finding the path to file
for dirpath, _, file_names in os.walk("data/"):
    for filename in file_names:
        print(os.path.join(dirpath, filename))


training = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
# print(training.head())

training['train_test'] = 1
test['train_test'] = 0
test['Survived'] = np.NaN
all_data = pd.concat([training,test])

print(all_data.columns)

# Take a closer look at the data given (data types and number of null values)
# print(training.info())
# print(training.describe())

# look at numeric and categorical values separately
df_num = training[['Age','SibSp','Parch','Fare']]
df_cat = training[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]

# for i in df_num.columns:
#     plt.hist(df_num[i])
#     plt.title(i)
#     plt.show()

# Find correlation between independent variables
# print(df_num.corr())
# sns.heatmap(df_num.corr(), annot=True)
# plt.show()

# compare survival rate across Age, SibSp, Parch, and Fare
print(pd.pivot_table(training, index = 'Survived', values = ['Age','SibSp','Parch','Fare']))
print()
# Look at the categorical data
# for i in df_cat.columns:
#     sns.barplot(df_cat[i].value_counts().index, df_cat[i].value_counts()).set_title(i)
#     plt.show()

# compare survival and each of the categorical variables
print(pd.pivot_table(training, index = 'Survived', columns='Pclass', values="Ticket", aggfunc='count'), "\n")
print(pd.pivot_table(training, index = 'Survived', columns='Sex', values="Ticket", aggfunc='count'), "\n")
print(pd.pivot_table(training, index = 'Survived', columns='Embarked', values="Ticket", aggfunc='count'), "\n")

# Preprocess training set: simplify Cabin column
# if that passenger has missing value for cabin then their cabin # is 0
# if that passenger has multiple cabins then save exactly how many they got
training['cabin_multiple'] = training.Cabin.apply(lambda x : 0 if pd.isna(x) else len(x.split(' ')))
# Vast majority don't have a cabin - implicating lots of missing values
# print(training['cabin_multiple'].value_counts())

# create categories based on the cabin letter
training['cabin_adv'] = training.Cabin.apply(lambda x: str(x)[0])
# print(training.cabin_adv.value_counts())
#comparing surivial rate by cabin
# print(pd.pivot_table(training,index='Survived',columns='cabin_adv', values = 'Name', aggfunc='count'))

#understand ticket values better
#numeric vs non numeric
training['numeric_ticket'] = training.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
training['ticket_letters'] = training.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)


# print(training['ticket_letters'].value_counts())

# difference of survival in numeric vs non-numeric tickets in survival rate
# print(pd.pivot_table(training, index="Survived", columns='numeric_ticket', values='Ticket', aggfunc='count'))

# survival rate across different ticket types
# See no significance
print(pd.pivot_table(training,index='Survived',columns='ticket_letters', values = 'Ticket', aggfunc='count'))

# preprocess the name of each person on board to get the number of females and males
print(training.Name.head(5))
training['name_title'] = training.Name.apply(lambda x: x.split(",")[1].split(".")[0].strip())
print(training['name_title'].value_counts())

# drop rows with missing value in 'embarked'. Only two instances of this in training and 0 in test data
all_data.dropna(subset=['Embarked'],inplace = True)

#create all categorical variables that we did above for both training and test sets
all_data['cabin_multiple'] = all_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
all_data['cabin_adv'] = all_data.Cabin.apply(lambda x: str(x)[0])
all_data['numeric_ticket'] = all_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
all_data['ticket_letters'] = all_data.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)
all_data['name_title'] = all_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

#impute nulls for continuous data
#all_data.Age = all_data.Age.fillna(training.Age.mean())
all_data.Age = all_data.Age.fillna(training.Age.median())
#all_data.Fare = all_data.Fare.fillna(training.Fare.mean())
all_data.Fare = all_data.Fare.fillna(training.Fare.median())

# Tried log normalization of sibsp (Not Used since it is not closer to normal distribution)
all_data['norm_sibsp'] = np.log(all_data.SibSp+1)
all_data['norm_sibsp'].hist()
# plt.show()

# Tried log normalization of fare (Used)
all_data['norm_fare'] = np.log(all_data.Fare+1)
all_data['norm_fare'].hist()
# plt.show()

all_data.Pclass = all_data.Pclass.astype(str)
print(all_data.Pclass.head(10))


# created dummy variables from categories
all_dummies = pd.get_dummies(all_data[['Pclass','Sex','Age','SibSp','Parch','norm_fare','Embarked','cabin_adv','cabin_multiple','numeric_ticket','name_title','train_test']])


# Split to train and test data
x_train = all_dummies[all_dummies.train_test == 1].drop(['train_test'], axis =1)
x_test = all_dummies[all_dummies.train_test == 0].drop(['train_test'], axis =1)
y_train = all_data[all_data.train_test==1].Survived
print(y_train.shape)


# Scale all the numerical variables to range [-1,1] using Standard Scaler
scale = StandardScaler()
all_dummies_scaled = all_dummies.copy()
all_dummies_scaled[['Age','SibSp','Parch','norm_fare']] = scale.fit_transform(all_dummies_scaled[['Age','SibSp','Parch','norm_fare']])
print(all_dummies_scaled)
x_train_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 1].drop(['train_test'], axis =1)
x_test_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 0].drop(['train_test'], axis =1)

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Naive Bayes (72.6%)
gnb = GaussianNB()
cv = cross_val_score(gnb,x_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())

# Logistic Regression (82.1%)
lr = LogisticRegression(max_iter = 2000)
cv = cross_val_score(lr,x_train,y_train,cv=5)
print(cv)
print(cv.mean())

# Decision Tree (77.6%)
dt = tree.DecisionTreeClassifier(random_state = 1)
cv = cross_val_score(dt,x_train,y_train,cv=5)
print(cv)
print(cv.mean())