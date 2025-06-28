import numpy as np
import pandas as pd 
import seaborn as sn 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
print(train.head())
print(train.describe())
print(train.info())
print(train.isnull().sum())
#here in the train dataset we find that the column age and cabin
#contains highly null and value and embarked also contains two null values so we need to remove those later.

print(test.head())
print(test.describe())
print(test.info())
print(test.isnull().sum())
#Now lets check how many people survied and how many dies 0 here represents died people and 1 represents the survival people
#and also the percentage of their survival. We can clearly see that 38% of people survived and 61% percent people died in the incident.
print(train['Survived'].value_counts())
print(train['Survived'].value_counts(normalize=True)*100)
plt.title("survived count",fontsize=16,color='red')
sn.countplot(x='Survived',data=train,palette='Set2')
plt.show()
#Now let's see the survival count according to the gender
print(train.groupby('Sex')['Survived'].value_counts())
plt.title("survived count according to gender",fontsize=16,color='blue')
sn.countplot(x='Sex',data=train,hue='Survived',palette='pastel')
plt.show()
# We can clearly see that the survived chance of women is more than that of the man..Maybe they are given the first priority than the men
print(train.groupby('Pclass')['Survived'].value_counts())
plt.title("survived count according to people class",fontsize=16,color='blue')
sn.countplot(x='Pclass',hue='Survived',data=train,palette="Set1")
plt.show()
#surivied count according to the people class
plt.title("survived count according to people class",fontsize=16,color='blue')
sn.violinplot(x='Pclass',y='Age',hue='Survived',split=True,data=train,palette="Set1")
plt.show()
#Distribution of the survived and death people accordance to their age and their classes.
#From the graph it seems like most of the pople belonging to second and third class of the 20 to 30 age died and in the first class the death count is so less those who died are mostly from the age group of 40 to 50.

#let's fill the empty value with some numeric value there is empty value mainly in Age, cabin and embarked which need to fill up
train['Age'].fillna(train['Age'].median(),inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0],inplace=True)
print(train.isnull().sum())

#test.csv
test['Age'].fillna(test['Age'].median(),inplace=True)
test.drop(['Cabin','Ticket'],axis=1,inplace=True)
test['Fare'].fillna(test['Fare'].median(),inplace=True)
print(test.isnull().sum())





plt.figure(figsize=(8, 5))
sn.kdeplot(data=train, x='Age', hue='Survived', fill=True, common_norm=False, palette='Set2', alpha=0.6)
plt.title('Age Distribution by Survival (KDE)', fontsize=14)
plt.xlabel('Age')
plt.ylabel('Density')
plt.grid(True)
plt.tight_layout()
plt.show()


#heat correlation graph

plt.figure(figsize=(10,8))
sn.heatmap(train.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()






#here i see some non-importance column that has no connection with prediction i am planning to drop those colums
train.drop(['Cabin','Ticket'],axis=1,inplace=True)
print(train.columns)
#times to perform some feature engineering and generate some new columns so that it helps model to predict more stuffs

train['f_size']=train['SibSp']+train['Parch']+1
train['isalone']=0
train.loc[train['f_size']==1,'isalone']=1
print(train[['SibSp','Parch','f_size','isalone']])

test['f_size']=test['SibSp']+test['Parch']+1
test['isalone']=0
test.loc[test['f_size']==1,'isalone']=1
print(test[['SibSp','Parch','f_size','isalone']])

print(train.groupby('isalone')['Survived'].value_counts())
plt.title("survived count according to gender",fontsize=16,color='blue')
sn.countplot(x='isalone',data=train,hue='Survived',palette='pastel')
plt.show()


train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
rare_titles=['Col','Mlle','Major','Ms',"Mme",'Don','Lady','Sir','Capt','Countess','Jonkheer','Dona']
test['Title']=test['Title'].replace(rare_titles,"Rare")

# Show counts of each title
print(train['Title'].value_counts())
rare_titles=['Col','Mlle','Major','Ms',"Mme",'Don','Lady','Sir','Capt','Countess','Jonkheer']
train['Title']=train['Title'].replace(rare_titles,"Rare")
print(train['Title'].value_counts())

print(train.groupby('Title')['Survived'].value_counts())

title_map = {
    'Mr': 1,
    'Miss': 2,
    'Mrs': 3,
    'Master': 4,
    'Dr': 5,
    'Rev': 6,
    'Rare': 7
}
train['Title_encoded'] = train['Title'].map(title_map)
print(train)
test['Title_encoded'] = test['Title'].map(title_map)
print(test)
print(test['Title'].value_counts())
print(test.isnull().sum())

plt.title("survived count according to title name of the people",fontsize=16,color='blue')
sn.countplot(x='Title',data=train,hue='Survived',palette='Set2')
plt.show()

g=sn.catplot(x='Sex',y='Age',hue='Survived',data=train,kind='box',height=4,aspect=0.8,col='Pclass')
g.fig.suptitle("Raw Information of People by Sex and Class (Age Distribution)", 
               fontsize=16, color='blue')

# Adjust spacing so title doesn't overlap
g.fig.subplots_adjust(top=0.85)
plt.show()

print(train.isnull().sum())


#let's see teh data analysis according to teh embarkment allocated to the people
print(train.groupby('Embarked')['Survived'].value_counts())

sn.violinplot(x='Embarked',y='Pclass',data=train,hue='Survived',split=True,palette='Set2',inner=None)
plt.show()



# #again let's do some feature engineering.
print(train['Fare'].describe())
train['FareBin'] = pd.qcut(train['Fare'], 4, labels=[0, 1, 2, 3])
print(train['FareBin'].value_counts())
title_map2={
    'C':0,
    'Q':1,
    'S':2
}
train['embark']=train['Embarked'].map(title_map2).astype(int)
print(train.columns)
le=LabelEncoder()
train['Sex']=le.fit_transform(train['Sex'])

test['FareBin'] = pd.qcut(test['Fare'], 4, labels=[0, 1, 2, 3])
print(test['FareBin'].value_counts())
title_map2={
    'C':0,
    'Q':1,
    'S':2
}
test['embark']=test['Embarked'].map(title_map2).astype(int)
print(test.columns)
le=LabelEncoder()
test['Sex']=le.fit_transform(test['Sex'])


def age_group(age):
    if age <= 4:
        return 'Baby'
    elif age <= 12:
        return 'Child'
    elif age <= 18:
        return 'Teen'
    elif age <= 25:
        return 'YoungAdult'
    elif age <= 60:
        return 'Adult'
    else:
        return 'Senior'

# Apply function
train['AgeGroup'] = train['Age'].apply(age_group)
test['AgeGroup'] = test['Age'].apply(age_group)
le_age = LabelEncoder()

train['AgeGroup'] = le_age.fit_transform(train['AgeGroup'])
print(train)
print(train.columns)
# train.to_csv('final.csv')
train.drop(['PassengerId','Name','Age','SibSp','Parch','Fare','Embarked','f_size','Title'],inplace=True,axis=1)
print(train.head())


test['AgeGroup'] = test['Age'].apply(age_group)
test['AgeGroup'] = test['Age'].apply(age_group)


test['AgeGroup'] = le_age.fit_transform(test['AgeGroup'])

print(test.columns)

test.drop(['Name','Age','SibSp','Parch','Fare','Embarked','f_size','Title'],inplace=True,axis=1)
print(test.head())


# #now it's time to really predict the data

x=train.drop('Survived',axis=1)
y=train['Survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=66)
models = [
    LogisticRegression(max_iter=1000),
    LinearSVC(max_iter=10000),
    SVC(kernel='rbf'),
    KNeighborsClassifier(),
    RandomForestClassifier(),
    DecisionTreeClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()
]
model_names = [
    'LogisticRegression',
    'LinearSVM',
    'rbfSVM',
    'KNearestNeighbors',
    'RandomForest',
    'DecisionTree',
    'GradientBoosting',
    'GaussianNB',
    'LinearDiscriminant',
    'QuadraticDiscriminant'
]
accuracy=[]
for clf in models:
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    acc = accuracy_score(y_test, pred)
    accuracy.append(acc)

# 🪞 Compare results
compare = pd.DataFrame({
    'Algorithm': model_names,
    'Accuracy': accuracy
}).sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

print(compare)

plt.figure(figsize=(10, 6))
sn.pointplot(data=compare, x='Accuracy', y='Algorithm', color='blue', markers='o', linestyles='-')
plt.title('Model Accuracy Comparison', fontsize=16)
plt.xlabel('Accuracy')
plt.ylabel('Model')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0.7, 0.9)  # Adjust based on your accuracy range
plt.tight_layout()
plt.show()

#trying tune some parameter to show is there any chance to increase the model accuracy more.
best_model = GradientBoostingClassifier()
best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sn.heatmap(cm, annot=True, fmt="d", cmap='YlGnBu')
plt.title('Confusion Matrix (Gradient Boosting)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


x = train.drop('Survived', axis=1)
y = train['Survived']

final_model = GradientBoostingClassifier()
final_model.fit(x, y)

features=['Pclass','Sex','isalone','Title_encoded','FareBin','embark','AgeGroup']
x_final=test[features]
f_predication=final_model.predict(x_final)

submission=pd.DataFrame({
    "PassengerId":test['PassengerId'],
    "Survived":f_predication
})
submission.to_csv("Submission.csv",index=False)
print("hey the submission file has been created")




