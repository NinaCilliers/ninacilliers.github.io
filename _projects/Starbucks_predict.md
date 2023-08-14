<h1>Predicting brand loyality </h1>
<h4><i>Optimizing an adaptive boosting classifier to preduct return visits to Starbucks</i></h4>

<h2>Introduction</h2>

A small survey was conducted in Malaysia to learn more about consumer behavior at Starbucks. In sum, a total of 122 individuals answered 20 questions. Questions were asked targeting current purchasing behavior as well as perceptions of Starbucks' products and facilities. Basic demographic information was also collected. Respondents were asked if they planned on returning to Starbucks in the future, which is taken as an indicator of consumer loyalty. 

We extensively explored this data and used clustering techniques to segment the Starbucks' consumer base in a {previous project](link). Here, a model to predict consumer loyalty was developed. Common classifiers in the Scikit learn package were sampled and optimized. Several classifiers individually reached ~87.5% accuracy on the test dataset. Accuracy was not improved by combining classifiers into a voting classifier. 

Finally, feature selection was performed using a random forest classifier. It is shown that customer loyalty is dependent on customers being able to afford Starbucks, perceiving Starbucks as high quality, and enjoying the ambiance. Loyalty prediction using just eight input features performed as well as all individual classifiers built on the full dataset. This indicates that these metrics should be the focus for loyalty improvement and future surveys.

<h2><br></h2>
<h2>Data cleaning</h2>


```python
#importing packages 
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import set_config
import seaborn as sns
from copy import deepcopy
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
#importing packages
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle

#importaing packages
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import loguniform,expon, uniform, randint
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
```


```python
#changing settings
set_config(transform_output="pandas")
pd.set_option('display.max_columns', 100)
plt.style.use('ggplot')
```


```python
with open('cleaned_data.pkl','wb') as f:
        pickle.dump(df,f)

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Income</th>
      <th>Frequency</th>
      <th>Duration</th>
      <th>Distance</th>
      <th>Price</th>
      <th>Quality rating</th>
      <th>Price rating</th>
      <th>Sale importance</th>
      <th>Ambiance rating</th>
      <th>Wifi rating</th>
      <th>Service rating</th>
      <th>Referral score</th>
      <th>Gender_Male</th>
      <th>Job_Employed</th>
      <th>Job_Housewife</th>
      <th>Job_Self-employed</th>
      <th>Job_Student</th>
      <th>Consumption Location_Dine in</th>
      <th>Consumption Location_Drive-thru</th>
      <th>Consumption Location_Never</th>
      <th>Consumption Location_Take away</th>
      <th>Member_Yes</th>
      <th>Future visits_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.333333</td>
      <td>0.00</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.333333</td>
      <td>0.75</td>
      <td>0.50</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.75</td>
      <td>0.75</td>
      <td>0.50</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.333333</td>
      <td>0.00</td>
      <td>0.25</td>
      <td>0.00</td>
      <td>0.5</td>
      <td>0.333333</td>
      <td>0.75</td>
      <td>0.50</td>
      <td>0.75</td>
      <td>0.75</td>
      <td>0.75</td>
      <td>1.00</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.333333</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.25</td>
      <td>1.0</td>
      <td>0.333333</td>
      <td>0.75</td>
      <td>0.50</td>
      <td>0.75</td>
      <td>0.75</td>
      <td>0.75</td>
      <td>0.75</td>
      <td>0.50</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.333333</td>
      <td>0.00</td>
      <td>0.25</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>0.333333</td>
      <td>0.25</td>
      <td>0.00</td>
      <td>0.75</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.333333</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.25</td>
      <td>0.5</td>
      <td>0.666667</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.75</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>117</th>
      <td>1.000000</td>
      <td>0.25</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.5</td>
      <td>0.666667</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>1.00</td>
      <td>0.50</td>
      <td>0.25</td>
      <td>0.75</td>
      <td>0.75</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>118</th>
      <td>0.333333</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.5</td>
      <td>1.000000</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>119</th>
      <td>0.333333</td>
      <td>0.00</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.5</td>
      <td>0.333333</td>
      <td>0.50</td>
      <td>0.25</td>
      <td>0.75</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.75</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>120</th>
      <td>0.333333</td>
      <td>0.00</td>
      <td>0.25</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.333333</td>
      <td>0.75</td>
      <td>0.75</td>
      <td>0.75</td>
      <td>0.75</td>
      <td>0.75</td>
      <td>0.75</td>
      <td>0.75</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>121</th>
      <td>0.333333</td>
      <td>0.50</td>
      <td>0.25</td>
      <td>0.25</td>
      <td>0.5</td>
      <td>0.333333</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.75</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.25</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>122 rows × 24 columns</p>
</div>



Summary of cleaned dataset:


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 122 entries, 0 to 121
    Data columns (total 24 columns):
     #   Column                           Non-Null Count  Dtype  
    ---  ------                           --------------  -----  
     0   Age                              122 non-null    float64
     1   Income                           122 non-null    float64
     2   Frequency                        122 non-null    float64
     3   Duration                         122 non-null    float64
     4   Distance                         122 non-null    float64
     5   Price                            122 non-null    float64
     6   Quality rating                   122 non-null    float64
     7   Price rating                     122 non-null    float64
     8   Sale importance                  122 non-null    float64
     9   Ambiance rating                  122 non-null    float64
     10  Wifi rating                      122 non-null    float64
     11  Service rating                   122 non-null    float64
     12  Referral score                   122 non-null    float64
     13  Gender_Male                      122 non-null    float64
     14  Job_Employed                     122 non-null    float64
     15  Job_Housewife                    122 non-null    float64
     16  Job_Self-employed                122 non-null    float64
     17  Job_Student                      122 non-null    float64
     18  Consumption Location_Dine in     122 non-null    float64
     19  Consumption Location_Drive-thru  122 non-null    float64
     20  Consumption Location_Never       122 non-null    float64
     21  Consumption Location_Take away   122 non-null    float64
     22  Member_Yes                       122 non-null    float64
     23  Future visits_Yes                122 non-null    float64
    dtypes: float64(24)
    memory usage: 23.0 KB
    

<h2><br></br>
<h2>Feature correlation</h2>
A full EDA and exploration of this data is available in a previous [project](). We present the feature correlation map here, as it guides what we should expect to seee in our model.

- The tendency to recommend Starbucks as a meeting place, quality rating, ambience rating, service rating, and the desire to visit Starbucks in the future are all positively correlated. 

- We also see that items in the consumption location and career categories are negatively correlated, as customers can only select one of these sub-categories. 

- Starbucks members tend to have high incomes, visit Starbucks more frequently, spend more money, and positively view Starbucks' quality. 

- Frequent visitors of Starbucks tend to spend more money and positively perceive the product quality. 

<h2><br></h2>
<h2>Selecting base classifiers</h2>


```python
from sklearn.model_selection import cross_val_score
```


```python
#Making test and training sets to predict Future visit score
X_K,y = df_unscaled.drop(columns=['Future visits_Yes']), df_unscaled['Future visits_Yes']

train_set, test_set = train_test_split(df_unscaled, test_size=0.3, stratify=df_unscaled['Future visits_Yes'], random_state=10)

X_train, y_train = train_set.drop(columns=['Future visits_Yes']), train_set['Future visits_Yes']
X_test, y_test = test_set.drop(columns=['Future visits_Yes']), test_set['Future visits_Yes']
```


```python
#Sampling selected classifiers
rnd_clf = RandomForestClassifier(n_estimators=50, oob_score=True, random_state=10)
gboost_clf = GradientBoostingClassifier(n_estimators=5000, n_iter_no_change=10, learning_rate = .7, random_state=10)
ada_clf = AdaBoostClassifier(n_estimators = 84, learning_rate=3.84, random_state=10)
extra_clf = ExtraTreesClassifier(n_estimators=15, random_state=10)
svc_clf = LinearSVC(random_state=10,C=100, max_iter=1000000)
svc_rbf_clf = SVC(kernel='rbf',C=18,gamma=.0071, max_iter=100000, random_state = 10,probability=True)
log_clf = LogisticRegression(C=.816,penalty='l1',solver='saga',random_state=10, max_iter=10000)
#LogisticRegression(C=.734,penalty='l1',solver='saga',random_state=10, max_iter=10000)

clfs = [rnd_clf, ada_clf, gboost_clf, extra_clf, svc_clf, svc_rbf_clf, log_clf]

acc = []
for classifier in clfs:
  classifier.fit(X_train, y_train)
  acc_ = cross_val_score(classifier,X_test, y_test,scoring='accuracy',cv=10,n_jobs=-1).mean()
  acc.append(acc_)

names = ['Random tree', 'Adaptive boosting', 'Gradiant boosting','Extra trees','Linear SVC', 'SVC rbf','Logistic regression']

plt.barh(names, acc)
plt.axvline(sum(acc)/len(acc), color='k', linestyle='--')
plt.title('Classifier accuracy score');
plt.show()
```


```python
pd.DataFrame(data=acc,index=names,columns=['Accuracy score (cv=10)']).sort_values('Accuracy score (cv=10)',ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy score (cv=10)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Adaptive boosting</th>
      <td>0.875000</td>
    </tr>
    <tr>
      <th>Logistic regression</th>
      <td>0.850000</td>
    </tr>
    <tr>
      <th>Random tree</th>
      <td>0.791667</td>
    </tr>
    <tr>
      <th>Gradiant boosting</th>
      <td>0.791667</td>
    </tr>
    <tr>
      <th>Extra trees</th>
      <td>0.791667</td>
    </tr>
    <tr>
      <th>SVC rbf</th>
      <td>0.725000</td>
    </tr>
    <tr>
      <th>Linear SVC</th>
      <td>0.683333</td>
    </tr>
  </tbody>
</table>
</div>



<h2><br></h2>
<h2>Optimizing base classifiers</h2>
Base clasifiers were optimzed through manual hyperparameter adjustment. The optimized parameters were updated in the above section such that the accuracy scores reflect the optimized model performance.


```python
iters = 1
rs = 10
```


```python
#Logistic regression optimization
log_best_clf = LogisticRegression(max_iter=100000, random_state=rs)
lin_params = [{
    'C':uniform(0.1,5),
    'solver':['saga'],
    'penalty':['l1']},{
  'C':uniform(0.1,5),
   'solver':['saga'],
   'penalty':['elasticnet'],
   'l1_ratio':uniform(.5,.9)},{   
    'C':uniform(.1,5),
   'solver':['liblinear'],
   'penalty':['l2']} 
]

rnd_search=RandomizedSearchCV(log_best_clf, param_distributions=lin_params, n_iter=iters*3, cv=10, scoring='accuracy',random_state=rs)
rnd_search.fit(X_train, y_train);
#Best model
%log_best_clf = LogisticRegression(C=.816,penalty='l1',solver='saga',random_state=10, max_iter=10000)
#best score 0.847
#rnd_search.best_score_, rnd_search.best_params_
```


```python
#Optimizing SVC
svc_best_clf =SVC(kernel='rbf', random_state=rs, max_iter=10000)

svc_params = {
    'C':randint(7,20),
    'gamma':uniform(0.001,0.05)
}

svc_search=RandomizedSearchCV(svc_best_clf, param_distributions=svc_params, n_iter=iters, cv=10, scoring='accuracy',random_state=rs)
svc_search.fit(X_train, y_train);
#best model 
#svc_best_clf =SVC(kernel='rbf', C=18, gamma=.0071, random_state=10, max_iter=10000)
#best score .844
#svc_search.best_score_, svc_search.best_params_
```


```python
#Optimizing Adaboost
ada_best_clf = AdaBoostClassifier(random_state=rs)
ada_params = {
   'n_estimators':randint(20,100),
   'learning_rate':uniform(1,5)}

ada_search=RandomizedSearchCV(ada_best_clf, param_distributions=ada_params, n_iter=iters, cv=10, scoring='accuracy', random_state=rs)
ada_search.fit(X_train, y_train);

#best model found 81.5%
#ada_base_best_clf = AdaBoostClassifier(n_estimators = 84, learning_rate=3.84)
#ada_search.best_score_, ada_search.best_params_
```


```python
#Optimizing Adaboost with base estimator 
ada_base_best_clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(), random_state=10)
ada_params = {
    'estimator__max_features':['sqrt', 'log2'],
   'estimator__max_depth':randint(1,20),
   'n_estimators':randint(1,200),
   'learning_rate':uniform(.1,50),
   'estimator__criterion': ['gini', 'entropy'],
   'estimator__min_weight_fraction_leaf':uniform(.1,.5)}

ada_search=RandomizedSearchCV(ada_base_best_clf, param_distributions=ada_params, n_iter=iters, cv=10, scoring='accuracy', random_state=rs)
ada_search.fit(X_train, y_train);

#best model found 
#ada_base_best_clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(min_weight_fraction_leaf=.11,criterion='entropy', max_depth=17, max_features='sqrt'),n_estimators=108, learning_rate = 25.7, random_state=10)
#best accuracy 89.1%, but performed the same as other adabase model without base estimator optimization on test set at ~84%
#ada_search.best_score_, ada_search.best_params_
```

Following previous analyses done on this data set, adaptive boosting was found to be particularly helpful in predicting customer loyalty during cross-validation with accuracy scores >89%, but failed to out-perform on the test set, suggesting over training or sampling issues. 

<h2><br></h2>
<h2>Ensemble classifiers</h2>
<h3>Voting classifier</h3>


```python
voting_clf = VotingClassifier(
    estimators=[('AdaBoost',ada_clf),
                ('ExtraTrees',extra_clf),
                ('Logistic',log_clf),
                ('GradBoost',gboost_clf)
                ],
                voting='hard'

)

voting_clf.fit(X_train,y_train)
acc_voting = cross_val_score(voting_clf,X_test, y_test,scoring='accuracy',cv=10,n_jobs=-1).mean()
acc_voting
```

    c:\Users\corne\anaconda3\lib\site-packages\sklearn\model_selection\_split.py:700: UserWarning: The least populated class in y has only 8 members, which is less than n_splits=10.
      warnings.warn(
    




    0.875



<h3><br></h3>
<h3>Stacking classifier</h3>


```python
stacking_clf = StackingClassifier(
    estimators=[('AdaBoost',ada_clf),
                ('Logistic',log_clf),
                ('GradBoost',gboost_clf),
                ('ExtraTrees',extra_clf)
                ],
                stack_method = 'predict_proba'

)

stacking_clf.fit(X_train,y_train)
acc_stacking = cross_val_score(voting_clf,X_test, y_test,scoring='accuracy',cv=10,n_jobs=-1).mean()
acc_stacking
```

    c:\Users\corne\anaconda3\lib\site-packages\sklearn\model_selection\_split.py:700: UserWarning: The least populated class in y has only 8 members, which is less than n_splits=10.
      warnings.warn(
    




    0.875



We fail to improve upon the accuracy of the adaptive boosting. <b>Thus, the best model for predicting customer loyalty from all the provided features is the adaptive boosting classifier.</b>

<h2><br></h2>
<h2>Feature importance</h2>

Key features for predicting consumer loyalty can be used by Starbucks to make improvements. It may also be possible to improve loyalty prediction accuracy.


```python
rnd_clf = RandomForestClassifier(n_estimators=50, oob_score=True, n_jobs=-1, random_state=10).fit(X_K, y)
rnd_clf.oob_score_
```




    0.8032786885245902




```python
feature_importance = pd.Series(rnd_clf.feature_importances_, index=rnd_clf.feature_names_in_).sort_values(ascending=False)
```


```python
plt.bar(feature_importance.index.values.tolist(), feature_importance, width=.7,color='lightblue')
plt.bar(feature_importance[:6].index.values.tolist(), feature_importance[:6], width=.7,color='crimson')
plt.ylabel('Feature importance', fontsize=14)
plt.xticks(rotation=90)
plt.show()
```


    
![png](output_31_0.png)
    


Key features for predicting loyalty include price rating, quality rating, proclivity to recommend Starbucks for a business or social meeting, ambiance rating, and duration of visit. 



<h2><br></h2>
<h2>Predicing loyalty using reduced features</h2>


```python
#Model definition
'''
ada_top_clf = AdaBoostClassifier(n_estimators=9, learning_rate=4.1, random_state=10)
log_top_clf = LogisticRegression(C=.816,penalty='l1',solver='saga',random_state=10, max_iter=10000)
svc_top_clf =SVC(kernel='rbf', random_state=10, max_iter=10000,C=18, gamma=.0071,probability=True)
'''

#selected_models = [ada_clf, log_clf, svc_clf]
#selected_names = ['Adaptive boosting','Logistic classifier','SVC classifier']

```


```python
#Finding the accuracy score for each feature number
score_ada = []
score_log = []
score_svc = []

scores = pd.DataFrame(data=np.zeros((22,len(names))),columns=names)
for kk in range(1,22):
  top_features = feature_importance[:kk]
  X_train_top = X_train[top_features.index.values.tolist()]
  X_test_top = X_test[top_features.index.values.tolist()]
  for m,model in enumerate(clfs):
    model.fit(X_train_top, y_train)
    acc_ = cross_val_score(model,X_test_top, y_test,scoring='accuracy',cv=10,n_jobs=-1).mean()
    scores.iloc[kk,m] = acc_
```


```python
#Plotitng results
x_range = range(0,22)

for n,model in enumerate(clfs):
    plt.plot(x_range, scores.iloc[:,n],'o-',label = names[n])
#lt.plot(x_range, scores['Logistic classifier'],'o-',label = 'Logistic regression', color='lightgreen')
#plt.plot(x_range, scores['SVC classifier'],'o-',label = 'SVC with rbf kernel', color='pink')
plt.legend(loc=4,facecolor='white')
plt.ylabel('Accuracy score')
plt.xlabel('Features selected')
plt.show()
```


    
![png](output_36_0.png)
    


We see the accuracy score increase as the feature number is increased from 0 to 8. There is no apparent advantage of adding more than 8 features, as the accuracy score is lower or the same from 8-21. 


```python
scores
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Random tree</th>
      <th>Adaptive boosting</th>
      <th>Gradiant boosting</th>
      <th>Extra trees</th>
      <th>Linear SVC</th>
      <th>SVC rbf</th>
      <th>Logistic regression</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.791667</td>
      <td>0.733333</td>
      <td>0.816667</td>
      <td>0.816667</td>
      <td>0.683333</td>
      <td>0.791667</td>
      <td>0.683333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.716667</td>
      <td>0.733333</td>
      <td>0.741667</td>
      <td>0.741667</td>
      <td>0.825000</td>
      <td>0.766667</td>
      <td>0.766667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.733333</td>
      <td>0.766667</td>
      <td>0.766667</td>
      <td>0.783333</td>
      <td>0.800000</td>
      <td>0.766667</td>
      <td>0.766667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.783333</td>
      <td>0.758333</td>
      <td>0.708333</td>
      <td>0.750000</td>
      <td>0.850000</td>
      <td>0.816667</td>
      <td>0.758333</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.708333</td>
      <td>0.758333</td>
      <td>0.683333</td>
      <td>0.850000</td>
      <td>0.850000</td>
      <td>0.816667</td>
      <td>0.758333</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.800000</td>
      <td>0.900000</td>
      <td>0.691667</td>
      <td>0.816667</td>
      <td>0.800000</td>
      <td>0.825000</td>
      <td>0.850000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.850000</td>
      <td>0.850000</td>
      <td>0.733333</td>
      <td>0.783333</td>
      <td>0.825000</td>
      <td>0.775000</td>
      <td>0.850000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.816667</td>
      <td>0.850000</td>
      <td>0.766667</td>
      <td>0.791667</td>
      <td>0.800000</td>
      <td>0.850000</td>
      <td>0.825000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.875000</td>
      <td>0.875000</td>
      <td>0.825000</td>
      <td>0.875000</td>
      <td>0.775000</td>
      <td>0.766667</td>
      <td>0.825000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.825000</td>
      <td>0.875000</td>
      <td>0.825000</td>
      <td>0.816667</td>
      <td>0.775000</td>
      <td>0.716667</td>
      <td>0.816667</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.850000</td>
      <td>0.875000</td>
      <td>0.825000</td>
      <td>0.791667</td>
      <td>0.750000</td>
      <td>0.750000</td>
      <td>0.816667</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.850000</td>
      <td>0.875000</td>
      <td>0.825000</td>
      <td>0.800000</td>
      <td>0.725000</td>
      <td>0.775000</td>
      <td>0.816667</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.791667</td>
      <td>0.875000</td>
      <td>0.825000</td>
      <td>0.800000</td>
      <td>0.750000</td>
      <td>0.800000</td>
      <td>0.816667</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.816667</td>
      <td>0.875000</td>
      <td>0.825000</td>
      <td>0.825000</td>
      <td>0.708333</td>
      <td>0.800000</td>
      <td>0.816667</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.816667</td>
      <td>0.875000</td>
      <td>0.791667</td>
      <td>0.775000</td>
      <td>0.691667</td>
      <td>0.775000</td>
      <td>0.816667</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.825000</td>
      <td>0.875000</td>
      <td>0.816667</td>
      <td>0.708333</td>
      <td>0.683333</td>
      <td>0.775000</td>
      <td>0.850000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.800000</td>
      <td>0.875000</td>
      <td>0.791667</td>
      <td>0.825000</td>
      <td>0.716667</td>
      <td>0.750000</td>
      <td>0.850000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.816667</td>
      <td>0.875000</td>
      <td>0.825000</td>
      <td>0.741667</td>
      <td>0.658333</td>
      <td>0.725000</td>
      <td>0.850000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.791667</td>
      <td>0.875000</td>
      <td>0.791667</td>
      <td>0.775000</td>
      <td>0.658333</td>
      <td>0.725000</td>
      <td>0.850000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.741667</td>
      <td>0.875000</td>
      <td>0.791667</td>
      <td>0.708333</td>
      <td>0.716667</td>
      <td>0.725000</td>
      <td>0.850000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.825000</td>
      <td>0.875000</td>
      <td>0.791667</td>
      <td>0.825000</td>
      <td>0.691667</td>
      <td>0.725000</td>
      <td>0.850000</td>
    </tr>
  </tbody>
</table>
</div>




```python
scores.max()
```




    Random tree            0.875
    Adaptive boosting      0.900
    Gradiant boosting      0.825
    Extra trees            0.875
    Linear SVC             0.850
    SVC rbf                0.850
    Logistic regression    0.850
    dtype: float64



We do not see a boost in performance from adding additional features to the model past feature 10. This indicates that future surveys can be brief and only collect this data. There is an increase in performance with the top six features and adaptive boosting. This classifier reaches 90% accuracy on the test set. However, this may be due to the limited number of available samples.



<h2><br></h2>
<h2>Conclusions</h2>

They key predictive features identified in this study indicate that affordability and quality are important factors for returning to Starbucks. The six key features that lead to improved single-classifier prediction of consumer loyalty were as follows: 

1. Positive perception of price options
2. Positive perception of product quality 
3. Recommending Starbucks as a business or social meeting place 
4. Perception of price 
5. Rating of ambiance 
6. Duration of visit 

To improve the likeliness of future visits to Starbucks, improving perceptions of affordability and ambiance would be the most helpful. All features are not necessary to predict the likeliness. In fact with an additional four features, listed below, the classifiers ae fully functional: 

7. Membership 
8. Frequency of visit 
9. Valuing promotions
10. Perception of service 

Collecting more data would enable a more robust statistical treatment of this data. Future surveys should focus on these 10 factors, as other factors were not found to be significant.
