import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

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

all_data.dropna(subset=['Embarked'],inplace = True)

#create all categorical variables that we did above for both training and test sets
all_data['cabin_multiple'] = all_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
all_data['cabin_adv'] = all_data.Cabin.apply(lambda x: str(x)[0])
all_data['numeric_ticket'] = all_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
all_data['ticket_letters'] = all_data.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)
all_data['name_title'] = all_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
