---
layout: page
title: Starbucks customer segmentation
description: Data reduction and clustering techniques are used to segment Starbucks consumers into distinct groups.
img: assets/img/starbucks_cluster/head.jpeg
importance: 5
category: Other projects
---

<h1>Starbucks consumer segmentation</h1>
<h4><i>Clustering consumer data into distinct segments</i></h4>
<h4><br></h4>
<h2>Introduction</h2>

A small survey was conducted in Malaysia to learn more about consumer behavior at Starbucks. In sum, a total of 122 individuals answered 20 questions. Questions were asked targeting current purchasing behavior as well as perceptions of Starbucks' products and facilities. Basic demographic information was also collected. Respondents were asked if they planned on returning to Starbucks in the future, which is taken as an indicator of consumer loyalty. 

Using this data, the customer base was segmented into groups. Cluster silhouette and inertia scores were used to determine the optimal number of consumer sub-groups. The most efficient cluster number suggested by these analyses (k=12) was too large given the limited data available, and the characteristics of formed groups was highly variable and changed with the random kernel used during clustering. 

Instead, a reduced number of clusters (k=5) was found to me more robust and offered clearer insight into consumer segments. It is likely that increasing the number of respondents in this data set would reveal a more complicated and potentially insightful map of consumer behavior. However, at k=5 we are still are able to segment the Starbucks' consumer base into meaningful groups. The five key groups identified were as follows:
<br>

1. Starbucks loyalists
2. Convenience shoppers
3. Drive thru users
4. Students on the go
5. Budget students

<h2><br></h2>
<h2>Project outline</h2>

1. Data cleaning
2. Feature correlation 
3. Cluster analysis 
<br> a. PCA
<br> b. Inertia scores
<br> c. Silhouette scores 
<br> d. Visualizing clusters (k=12)
<br> e. Visualizing clusters (k=5)
4. Customer segmentation analysis 
<br> a. Group loyalty
<br> b. Group demographics
<br> c. Group job type
<br> d. Group experience ratings
<br> e. Group dining location
<br> f. Group purchasing behavior 
<br> g. Group Identities
5. Recommendations 


<h2><br></h2>
<h2>Cleaning data</h2>

<details>
  <summary>Click to show hidden code.</summary>
  <pre>
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
  from sklearn.decomposition import PCA 
  from sklearn.cluster import KMeans
  from sklearn.metrics import silhouette_score
  import pickle
  </pre>
  <pre>
  set_config(transform_output="pandas")
  pd.set_option('display.max_columns', 100)
  plt.style.use('ggplot')
  </pre>
</details>

```python
def import_data(file_path):
    df = pd.read_csv(file_path)
    print("Survey resondants: ",df.shape[0])
    print("Questions asked: ", df.shape[1])
    questions = df.columns.values.tolist()
    return questions, df
```


```python
#renaming columns with simple names
def rename_cols(df):
    df = df.rename(columns={
        '1. Your Gender':'Gender',
        '2. Your Age':'Age',
        '3. Are you currently....?':'Job',
        '4. What is your annual income?':'Income',
        '5. How often do you visit Starbucks?':'Frequency',
        '6. How do you usually enjoy Starbucks?':'Consumption Location',
        '7. How much time do you normally  spend during your visit?':'Duration',
        '8. The nearest Starbucks\'s outlet to you is...?':'Distance',
        '9. Do you have Starbucks membership card?':'Member',
        '10. What do you most frequently purchase at Starbucks?':'Item',
        '11. On average, how much would you spend at Starbucks per visit?':'Price',
        '12. How would you rate the quality of Starbucks compared to other brands (Coffee Bean, Old Town White Coffee..) to be:':'Quality rating',
        '13. How would you rate the price range at Starbucks?':'Price rating',
        '14. How important are sales and promotions in your purchase decision?':'Sale importance',
        '15. How would you rate the ambiance at Starbucks? (lighting, music, etc...)':'Ambiance rating',
        '16. You rate the WiFi quality at Starbucks as..':'Wifi rating',
        '17. How would you rate the service at Starbucks? (Promptness, friendliness, etc..)':'Service rating',
        '18. How likely you will choose Starbucks for doing business meetings or hangout with friends?':'Referral score',
        '19. How do you come to hear of promotions at Starbucks? Check all that apply.':'Marketing list',
        '20. Will you continue buying at Starbucks?':'Future visits'
        })
    return df
```


```python
#Dropping columns that we are not going to consider in our analysis 
#Separate projects could be done on marketing or item choice 

def drop_cols(df):
    df = df.drop(columns=['Marketing list','Item','Timestamp'])
    return df
```


```python
#Cleaning Consumption Location
def clean_location(df):
    df['Consumption Location'] = df['Consumption Location'].replace({'never':'Never','Never buy':'Never','Never ':'Never','I dont like coffee':np.nan})
    return df
```


```python
#We are going to encode categorical data. 
def encode_cols(df):
    #Ordinal data is encoded using the OrdinalEncoder 
    age_enc = OrdinalEncoder(categories=[['Below 20','From 20 to 29','From 30 to 39','40 and above']])
    df['Age'] = age_enc.fit_transform(df[['Age']])

    income_enc = OrdinalEncoder(categories=[['Less than RM25,000', 'RM25,000 - RM50,000', 'RM50,000 - RM100,000','RM100,000 - RM150,000','More than RM150,000']])
    df['Income'] = income_enc.fit_transform((df[['Income']]))

    freq_enc = OrdinalEncoder(categories=[['Never','Rarely','Monthly','Weekly','Daily']])
    df['Frequency'] = freq_enc.fit_transform((df[['Frequency']]))

    dur_enc = OrdinalEncoder(categories =[['Below 30 minutes','Between 30 minutes to 1 hour','Between 1 hour to 2 hours','Between 2 hours to 3 hours','More than 3 hours']])
    df['Duration'] = dur_enc.fit_transform(df[['Duration']])

    dist_enc = OrdinalEncoder(categories=[['within 1km','1km - 3km','more than 3km']])
    df['Distance'] = dist_enc.fit_transform(df[['Distance']])

    price_enc = OrdinalEncoder(categories=[['Zero','Less than RM20','Around RM20 - RM40','More than RM40']])
    df['Price'] = price_enc.fit_transform(df[['Price']])
    
    #We encode non-ordinal categorical data as dummy variables 
    df = pd.get_dummies(df, columns=['Gender','Job','Consumption Location','Member','Future visits'])
    df = df.drop(columns = ['Gender_Female','Member_No','Future visits_No']) #We only need one column for binary categories
    return df
```


```python
def min_max(df):
    #Scaling data from 0 to 1 for each column for clustering
    df_unscaled = df.copy()

    scaler = MinMaxScaler() #This will work well as there are no outliers in dataset
    df = scaler.fit_transform(df)
    return df, df_unscaled
```


```python
class GetClean():
    def __init__(self, path):
        self.file_path = file_path

    def __call__(self):
        questions, df =  import_data(file_path)
        df = rename_cols(df)
        df = drop_cols(df)
        df = clean_location(df)
        df = encode_cols(df)
        df,df_unscaled = min_max(df)
        return df,df_unscaled,questions
```


```python
df, df_unscaled, questions = GetClean(file_path)()
```

    Survey resondants:  122
    Questions asked:  21
    


```python
questions
```




    ['Timestamp',
     '1. Your Gender',
     '2. Your Age',
     '3. Are you currently....?',
     '4. What is your annual income?',
     '5. How often do you visit Starbucks?',
     '6. How do you usually enjoy Starbucks?',
     '7. How much time do you normally  spend during your visit?',
     "8. The nearest Starbucks's outlet to you is...?",
     '9. Do you have Starbucks membership card?',
     '10. What do you most frequently purchase at Starbucks?',
     '11. On average, how much would you spend at Starbucks per visit?',
     '12. How would you rate the quality of Starbucks compared to other brands (Coffee Bean, Old Town White Coffee..) to be:',
     '13. How would you rate the price range at Starbucks?',
     '14. How important are sales and promotions in your purchase decision?',
     '15. How would you rate the ambiance at Starbucks? (lighting, music, etc...)',
     '16. You rate the WiFi quality at Starbucks as..',
     '17. How would you rate the service at Starbucks? (Promptness, friendliness, etc..)',
     '18. How likely you will choose Starbucks for doing business meetings or hangout with friends?',
     '19. How do you come to hear of promotions at Starbucks? Check all that apply.',
     '20. Will you continue buying at Starbucks?']




```python
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
    


```python
with open('cleaned_data.pkl','wb') as f:
    pickle.dump([df,df_unscaled],f)
```

# Feature correlation

To assess correlations between features in a dataset the Pearson correlation coefficient is calculated for each pair and visualized in a heatmap.

<details>
  <summary>Click to show hidden code.</summary>
  <pre>
  #Heat map 

  #optimizng range for color scale
  min = df.corr().min().min()
  max = df.corr()[df.corr()!=1].max().max()

  #thresholding selected correlations
  df_corr  = df.corr()[np.absolute(df.corr())>0.3]

  #Mask for selecting only bottom triangle
  mask = np.triu(df_corr)

  with plt.style.context('default'):
    sns.heatmap(df_corr, vmin=min, vmax=max, mask=mask)
    plt.show()
  </pre>
</details>

 
![png](/assets/img/starbucks_cluster/output_20_0.png)
    


- The tendency to recommend Starbucks as a meeting place, quality rating, ambience rating, service rating, and the desire to visit Starbucks in the future are all positively correlated. 

- We also see that items in the consumption location and career categories are negatively correlated, as customers can only select one of these sub-categories. 

- Starbucks members tend to have high incomes, visit Starbucks more frequently, spend more money, and positively view Starbucks' quality. 

- Frequent visitors of Starbucks tend to spend more money and positively perceive the product quality. 

<h2><br></h2>
<h2>Clustering survey data</h2>
We divide consumers into subset using the k-means clustering algorithm. The number of features in the dataset is first reduced using principal components analysis (PCA). Inertia and silhouette scores are used to determine the optimal cluster number.

Clusters are visualized against principal components. A reduced number of clusters is selected for functional purposes. Additional data needs to be added to meaningfully segment consumers into more groups.

<h3><br></h3>
<h3>Dimensionality reduction using PCA</h3>


```python
pca = PCA(n_components = 0.95,random_state=10)
X_pca = pca.fit_transform(df)
print('PCA reduced dimensions from ', df.shape[1],' to ',X_pca.shape[1] ,' and preserved 95% of variance.')
```

    PCA reduced dimensions from  24  to  15  and preserved 95% of variance.
    
<details>
  <summary>Click to show hidden code.</summary>
  <pre>
  plt.bar(range(pca.n_components_), pca.explained_variance_ratio_,color='mediumseagreen')
  plt.xlabel('Principle component')
  plt.ylabel('Explained variance ratio')
  #plt.xticks(range(pca.n_components_));
  plt.show()
  </pre>
</details>

    
![png](/assets/img/starbucks_cluster/output_25_0.png)
    


The explained variance ratio of each principal component is shown in the figure above. The first three principal components preserve ~45% of the variance. These principal components will be used to visualize clusters.
<h3><br></h3>
<h3>Selecting cluster number: Inertia and silouette scores</h3>

The inertia score is the sum of the squared distances between instances and their closest centroids. When selecting the optimal number of clusters for a dataset, the smallest number of clusters with a small inertia should be selected.


```python
#Inertia
k_range = np.arange(2,20,1)
kmeans = [KMeans(n_clusters = k, n_init=100, random_state=10).fit(X_pca) for k in k_range]
inertia = [kmeans.inertia_ for kmeans in kmeans]
```
<details>
  <summary>Click to show hidden code.</summary>
  <pre>
  plt.plot(k_range, inertia,'o-', color = 'royalblue')
  plt.plot([12], [inertia[10]],'x', color = 'yellow')
  #plt.axvline(x=2)
  plt.xlabel('k', fontsize=14)
  plt.ylabel('Inertia', fontsize=14)
  plt.show()
  </pre>
</details>
    
![png](/assets/img/starbucks_cluster/output_29_1.png)
    


The inertia scores for clustering features into 2-20 groups is shown above. k=12 is highlighted following as the maximum silhouette score. 

The silhouette score is the mean silhouette coefficient over all instances in a cluster. The silhouette coefficient considers the inter-cluster and nearest outer-cluster distance for each instance. The value can vary from -1 and 1 and is ideally 1.



```python
#sillouette score
labels = [model.labels_ for model in kmeans] #prints out cluster each point belongs to with k number of clusters defined
s_scores = [silhouette_score(df, kmeans.labels_) for kmeans in kmeans] #you need more than one cluster to calculate silhouette score

max_idx = np.argmax(s_scores)
max_k = k_range[max_idx]
max_score = s_scores[max_idx]
```

<details>
  <summary>Click to show hidden code.</summary>
  <pre>
  plt.plot(k_range, s_scores,'o-',color = 'royalblue')
  plt.plot(max_k, max_score,'x',c='yellow')
  plt.text(max_k+.5, max_score,f'Max k = {max_k}')
  plt.xlabel('k', fontsize=14)
  plt.ylabel('Silouette Scores', fontsize=14)
  plt.show()
  </pre>
</details>

![png](/assets/img/starbucks_cluster/output_31_0.png)
    


The silhouette score for cluster numbers from 2 to 20. From this analysis, k=12 is the optimal cluster number. There appears to be plateau silhouette score approached as cluster number increases past k=9.

<h2><br></h2>
<h2>Visualizing clusters</h2>
<h3>k=12</h3>


```python
#Adding cluster identity to original dataframe
max_k = 12
kmeans_best = deepcopy(kmeans[max_k-2])
X_pca['Cluster ID'] = kmeans_best.fit_predict(X_pca)
df['Cluster ID'] = X_pca['Cluster ID']
```

<details>
  <summary>Click to show hidden code.</summary>
  <pre>
  #plotting
  plt.figure(figsize=(10,8))
  plt.subplot(3,3,1)
  sns.scatterplot(data=X_pca, x='pca0', y='pca1', hue='Cluster ID',palette='Spectral', legend=False, s=15)
  plt.grid(False)
  plt.subplot(3,3,2)
  sns.scatterplot(data=X_pca, x='pca0', y='pca2', hue='Cluster ID',palette='Spectral', legend=False, s=15)
  plt.grid(False)
  plt.subplot(3,3,3)
  ax = plt.gca()
  sns.scatterplot(data=X_pca, x='pca1', y='pca2', hue='Cluster ID',palette='Spectral', legend=True, s=15)

  plt.tight_layout()
  ax.legend(title='Cluster ID',bbox_to_anchor=(1, 1),facecolor='white',edgecolor='k',fontsize=8)
  plt.show()
  </pre>
</details>

![png](/assets/img/starbucks_cluster/output_35_0.png)
    


We see diffuse clusters of consumer groups in the above graphs. There is a separation of groups apparent in pca2 vs. pca0. However, there is a high degree of overlap. For example, cluster 3 entirely overlaps cluster 4 in the above figures. Similarly, cluster 6 consistently overlaps multiple other clusters. 

Some of these twelve groups are helpful in isolating distinct groups of consumer behavior. For example, there are distinct groups for consumers that "Never" visit Starbucks as well as specific customer segments, like wealthy males who use the drive thru. These groups are lost in the k=5 cluster model. 

However, each group contains ~10 instances, which does not appear to be sufficient. Clustering with k=12 resulted in highly variable groupings dependent on the random kernel used. Pending the availability of more survey data, a lower number of clusters will be more appropriate and helpful.


<h3><br></h3>
<h3>k=5</h3>

Clusters are instead grouped into k=5, which through trial and error was found to produce the most consistant and distinct consumer groups.


```python
#Adding cluster identity to original dataframe
max_k = 5
kmeans_best = deepcopy(kmeans[max_k-2])
X_pca = X_pca.T[:-1].T #removes cluster ID from k=12 analysis above
X_pca['Cluster ID'] = kmeans_best.fit_predict(X_pca)
df['Cluster ID'] = X_pca['Cluster ID']
```

```python
#Re-ordering cluster numbers following tendoncy to want to visit Starbucks again in the future 
df_key = df.groupby('Cluster ID').agg('mean').sort_values('Future visits_Yes', ascending=False).reset_index().iloc[:,0]
df_key = df_key.to_dict()
df_key = {y: x for x, y in df_key.items()}
df.replace({'Cluster ID':df_key},inplace=True)
```

Mean features for k=5 clusters:
```python
df.groupby('Cluster ID').agg('mean').sort_values('Future visits_Yes', ascending=False).reset_index()
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
      <th>Cluster ID</th>
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
      <td>0</td>
      <td>0.444444</td>
      <td>0.305556</td>
      <td>0.388889</td>
      <td>0.287037</td>
      <td>0.537037</td>
      <td>0.530864</td>
      <td>0.796296</td>
      <td>0.638889</td>
      <td>0.805556</td>
      <td>0.814815</td>
      <td>0.564815</td>
      <td>0.768519</td>
      <td>0.768519</td>
      <td>0.592593</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.222222</td>
      <td>0.111111</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.851852</td>
      <td>0.925926</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.430108</td>
      <td>0.169355</td>
      <td>0.362903</td>
      <td>0.072581</td>
      <td>0.516129</td>
      <td>0.483871</td>
      <td>0.677419</td>
      <td>0.451613</td>
      <td>0.766129</td>
      <td>0.741935</td>
      <td>0.612903</td>
      <td>0.774194</td>
      <td>0.709677</td>
      <td>0.387097</td>
      <td>0.935484</td>
      <td>0.032258</td>
      <td>0.032258</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.967742</td>
      <td>0.612903</td>
      <td>0.903226</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.400000</td>
      <td>0.312500</td>
      <td>0.387500</td>
      <td>0.137500</td>
      <td>0.675000</td>
      <td>0.616667</td>
      <td>0.737500</td>
      <td>0.512500</td>
      <td>0.600000</td>
      <td>0.650000</td>
      <td>0.512500</td>
      <td>0.625000</td>
      <td>0.587500</td>
      <td>0.500000</td>
      <td>0.450000</td>
      <td>0.050000</td>
      <td>0.150000</td>
      <td>0.350000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.550000</td>
      <td>0.850000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.269841</td>
      <td>0.047619</td>
      <td>0.309524</td>
      <td>0.023810</td>
      <td>0.738095</td>
      <td>0.365079</td>
      <td>0.619048</td>
      <td>0.416667</td>
      <td>0.666667</td>
      <td>0.654762</td>
      <td>0.619048</td>
      <td>0.666667</td>
      <td>0.452381</td>
      <td>0.238095</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.238095</td>
      <td>0.761905</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.095238</td>
      <td>0.904762</td>
      <td>0.285714</td>
      <td>0.714286</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.333333</td>
      <td>0.097826</td>
      <td>0.206522</td>
      <td>0.173913</td>
      <td>0.804348</td>
      <td>0.304348</td>
      <td>0.478261</td>
      <td>0.326087</td>
      <td>0.597826</td>
      <td>0.532609</td>
      <td>0.489130</td>
      <td>0.543478</td>
      <td>0.554348</td>
      <td>0.608696</td>
      <td>0.217391</td>
      <td>0.000000</td>
      <td>0.086957</td>
      <td>0.695652</td>
      <td>0.826087</td>
      <td>0.0</td>
      <td>0.130435</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.391304</td>
    </tr>
  </tbody>
</table>
</div>



Visualizing clusters against principal components

<details>
  <summary>Click to show hidden code.</summary>
  <pre>
  #plotting k=5 clusters
  plt.figure(figsize=(10,8))
  plt.subplot(3,3,1)
  sns.scatterplot(data=X_pca, x='pca0', y='pca1', hue='Cluster ID',palette='Spectral', legend=False, s=15)
  plt.grid(False)
  plt.subplot(3,3,2)
  sns.scatterplot(data=X_pca, x='pca0', y='pca2', hue='Cluster ID',palette='Spectral', legend=False, s=15)
  plt.grid(False)
  plt.subplot(3,3,3)
  ax = plt.gca()
  sns.scatterplot(data=X_pca, x='pca1', y='pca2', hue='Cluster ID',palette='Spectral', legend=True, s=15)
  plt.grid(False)
  plt.tight_layout()
  ax.legend(title='Cluster ID',bbox_to_anchor=(1, 1),facecolor='white',edgecolor='k',fontsize=8)
  plt.show()
  </pre>
</details>
    
![png](/assets/img/starbucks_cluster/output_45_0.png)
    

* We see less overlap between clusters and larger groups with ~2x more instances per group. 
*   Diffuse clusters are apparent in pca1 vs. pca2 and pca 2 vs. pca1. 
*   Two seperable groups are apparent in pca0 vs. pca2. These findings are summarized in the table below. 

The separation apparent in pca2 vs pca1 is mostly driven by consumption location, with the top group having as strong preferance for taking their order to go, while the bottom group tends to use the drive through or has never been to a Starbucks. Other differences are summarized in the table below:

<details>
  <summary>Click to show hidden code.</summary>
  <pre>
  pca_avg = df.groupby('Cluster ID').agg('mean').reset_index()
  top_dict = {11:0, 10:0, 7:0, 2:0, 0:0,9:1,8:1,6:1,5:1,4:1,3:1,1:1}
  pca_avg['Cluster'] = pca_avg['Cluster ID'].replace(top_dict)
  pca_avg = pca_avg.groupby('Cluster').agg('mean')
  pca_avg = pca_avg.drop(columns=['Cluster ID']).T.reset_index()
  pca_avg['%'] = np.absolute(pca_avg[0]-pca_avg[1])/np.mean([pca_avg[0],pca_avg[1]])*100
  pca_avg = pca_avg.set_index('index')
  pca_avg['%'] = pca_avg['%'].astype(int)
  pca_avg[[0,1]]=pca_avg[[0,1]].round(2)
  pca_avg.rename(columns={0:'Top',1:'Bottom'}, inplace=True)
  pca_avg = pca_avg[pca_avg['%']>40].sort_values('%', ascending=False)
  pca_avg
  </pre>
</details>




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
      <th>Cluster</th>
      <th>Top</th>
      <th>Bottom</th>
      <th>%</th>
    </tr>

  </thead>
  <tbody>
    <tr>
      <th>Consumption Location_Take away</th>
      <td>0.00</td>
      <td>0.62</td>
      <td>144</td>
    </tr>
    <tr>
      <th>Consumption Location_Drive-thru</th>
      <td>0.50</td>
      <td>0.00</td>
      <td>115</td>
    </tr>
    <tr>
      <th>Member_Yes</th>
      <td>0.70</td>
      <td>0.31</td>
      <td>89</td>
    </tr>
    <tr>
      <th>Job_Student</th>
      <td>0.23</td>
      <td>0.49</td>
      <td>59</td>
    </tr>
    <tr>
      <th>Consumption Location_Dine in</th>
      <td>0.50</td>
      <td>0.28</td>
      <td>52</td>
    </tr>
    <tr>
      <th>Future visits_Yes</th>
      <td>0.89</td>
      <td>0.67</td>
      <td>50</td>
    </tr>
    <tr>
      <th>Income</th>
      <td>0.31</td>
      <td>0.10</td>
      <td>47</td>
    </tr>
    <tr>
      <th>Price</th>
      <td>0.57</td>
      <td>0.38</td>
      <td>43</td>
    </tr>
    <tr>
      <th>Price rating</th>
      <td>0.58</td>
      <td>0.40</td>
      <td>41</td>
    </tr>
  </tbody>
</table>
</div>


<h2><br></h2>
<h2>Consumer segmentation analysis</h2>
<h3>Group Loyalty</h3>

<details>
  <summary>Click to show hidden code.</summary>
  <pre>
  #Fraction of customers that plan on returning to Starbucks in each cluster
  f_visit_counts = df.groupby(['Future visits_Yes','Cluster ID']).size().reset_index()
  f_visit_counts = f_visit_counts[f_visit_counts['Future visits_Yes']==1]
  f_visit_counts = f_visit_counts.reset_index(drop=True)[0]/df.groupby('Cluster ID').size()
  f_visit_counts = f_visit_counts.reset_index()
  </pre>
  <pre>
  #Visualizing
  sns.barplot(data=f_visit_counts, y=0, x='index', color='dodgerblue');
  plt.xlabel('Cluster ID', fontsize=14)
  plt.ylabel('Returning customer fraction', fontsize=14)
  plt.show()
  </pre>
</details>
    
![png](/assets/img/starbucks_cluster/output_50_0.png)
    


Groups 0-2 are very likely to return to Starbucks, while 3 and 4 are less likely to return.


<h3><br></h3>
<h3>Group demographics</h3>
<details>
  <summary>Click to show hidden code.</summary>
  <pre>
  #Cluster demographics 
  fig = plt.figure(figsize=(8,4))
  plt.subplot(231)
  sns.boxplot(data=df, x='Cluster ID', y='Age',palette='Pastel1')
  plt.subplot(232)
  sns.boxplot(data=df, x='Cluster ID', y='Income',palette='Pastel1')
  plt.subplot(233)
  sns.boxplot(data=df, x='Cluster ID', y='Gender_Male',palette='Pastel1')
  plt.subplot(234)
  sns.boxplot(data=df, x='Cluster ID', y='Distance',palette='Pastel1')
  plt.tight_layout()
  plt.show()
  </pre>
</details>
    
![png](/assets/img/starbucks_cluster/output_53_0.png)
    



*   Groups have similar median ages. Groups 0-2 are skewed towards older ages.
*  Groups 0-2 median income is higher than clusters 3-4, which have median values close to zero. 
* Groups are mostly mixed gender, but group 3 is mostly female.
* Groups 2-4 live farther from a Starbucks.


<h3><br></h3>
<h3>Group job type</h3>

<details>
  <summary>Click to show hidden code.</summary>
  <pre>
  job_pie = df[['Job_Employed','Job_Self-employed','Job_Student','Job_Housewife','Cluster ID']].groupby('Cluster ID').agg('sum')
  labels = ['Employed','Self-\nEmployed','Student','Housewife']
  colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
  fig = plt.figure(figsize=(10,10))
  for row in range(job_pie.shape[0]):
    plt.subplot(1,5,row+1)
    plt.title(f'Cluster {row}')
    pie_data = job_pie.T[job_pie.T != 0][row].fillna(0)
    #pie_label = [labels[i] if pie_data[i]!=0 else '' for i in range(0,4) ]
    pie_label = [labels[i] for i in range(0,4) if pie_data[i]!=0]
    pie_color = [colors[i] for i in range(0,4) if pie_data[i]!=0]
    pie_data = job_pie.T[job_pie.T != 0][row].dropna()
    plt.pie(pie_data, labels=pie_label, autopct='%.1f%%',colors=pie_color, textprops={'size': 'x-small'}, startangle=90)
  plt.show()
  </pre>
</details>
    
![png](/assets/img/starbucks_cluster/output_56_0.png)
    


* Groups 0 and 1 are mostly employed.
* Groups 3 and 4 are mostly students. 
* Group 2 is mixed.


<h3><br></h3>
<h3>Group experience ratings</h3>

<details>
  <summary>Click to show hidden code.</summary>
  <pre>
  fig = plt.figure(figsize=(8,4))
  plt.subplot(231)
  sns.boxplot(data = df, y='Quality rating', x='Cluster ID',palette='Pastel1')
  plt.subplot(232)
  sns.boxplot(data = df, y='Price rating', x='Cluster ID',palette='Pastel1')
  plt.subplot(233)
  sns.boxplot(data = df, y='Wifi rating', x='Cluster ID',palette='Pastel1')
  plt.subplot(234)
  sns.boxplot(data = df, y='Service rating', x='Cluster ID',palette='Pastel1')
  plt.subplot(235)
  sns.boxplot(data = df, y='Ambiance rating', x='Cluster ID',palette='Pastel1')
  plt.subplot(236)
  sns.boxplot(data = df, y='Referral score', x='Cluster ID',palette='Pastel1')
  plt.tight_layout()
  plt.show()
  </pre>
</details>
      
![png](/assets/img/starbucks_cluster/output_59_1.png)
    

* Ratings tend to decrease as groups become less likely to visit Starbucks in the future. 
* Wi-Fi ratings are an exception to this, as it remains constant across clusters. It is likely that this Starbucks does not have issues with their Wi-Fi.
* Improving these metrics may lead to improved customer loyalty.

<h3><br></h3>
<h3>Visit type</h3>

<details>
  <summary>Click to show hidden code.</summary>
  <pre>
  loc_pie = df[['Consumption Location_Dine in','Consumption Location_Drive-thru','Consumption Location_Never','Consumption Location_Take away','Cluster ID']].groupby('Cluster ID').agg('sum')
  labels = ['Dine in','Drive-thru','Never','Take away']
  colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
  fig = plt.figure(figsize=(10,10))
  for row in range(job_pie.shape[0]):
    plt.subplot(1,5,row+1)
    plt.title(f'Cluster {row}')
    pie_data = loc_pie.T[loc_pie.T != 0][row].fillna(0)
    #pie_label = [labels[i] if pie_data[i]!=0 else '' for i in range(0,4) ]
    pie_label = [labels[i] for i in range(0,4) if pie_data[i]!=0]
    pie_color = [colors[i] for i in range(0,4) if pie_data[i]!=0]
    pie_data = loc_pie.T[loc_pie.T != 0][row].dropna()
    plt.pie(pie_data, labels=pie_label, autopct='%.1f%%',colors=pie_color, textprops={'size': 'x-small'}, startangle=90)
  plt.show()
  </pre>
</details>
    
![png](/assets/img/starbucks_cluster/output_62_0.png)
    

* Groups 0 and 4 dine in. 
* Groups 1 and 3 take away. 
* Group 2 drives through. 

<h3><br></h3>
<h3>Group purchasing behavior</h3>

<details>
  <summary>Click to show hidden code.</summary>
    <pre>
  #Consumer profiles 
  fig = plt.figure(figsize=(8,4))
  plt.subplot(231)
  sns.boxplot(data = df, y='Price', x='Cluster ID',palette='Pastel1')
  plt.subplot(232)
  sns.boxplot(data = df, y='Sale importance', x='Cluster ID',palette='Pastel1')
  plt.subplot(233)
  sns.boxplot(data = df, y='Member_Yes', x='Cluster ID',palette='Pastel1')
  plt.tight_layout()
  plt.show()
  </pre>
</details>

![png](/assets/img/starbucks_cluster/output_66_0.png)
    


*   Group 4 buys cheaper items than the other groups. This group may represent students with little disposable income.
*   Sales and promotions are important to everyone. 
*   The most likely group to return to Starbucks is made up of Starbucks members, while the least likely is not.  


<h2>Consumer Group Identity </h2>
From the above analysis we name and describe each group:

**Group 1: Starbucks loyalists**<br>
This group has the best regard for price options, product quality, and ambiance. They are the most likely to recommend Starbucks as a meeting place. They work and have the highest median income. They dine in and enjoy wholistically enjoy the Starbucks' experience.


**Group 2: Convenience shoppers**<br> 
This group has overall positive regard for Starbucks' price options, quality and ambiance. They are employed and take their products to go. While still likely to return to Starbucks, this group does not appear to regard the experience with as much enthusiasm and may instead be more product focused, viewing Starbucks as a pleasant convenience. 
  
**Group 3: Dive-through users**<br>
This mixed group is unified in their preference for the Starbucks' drive through. They tend to be less enthsiastic than the starbucks loyalists and convenience shoppers but still positively regard Starbucks and are still likely to visit in the future. 

**Group 4: Students on the go**<br> 
This is the most distinct group identified and is composed of female student who take their Starbucks to go. While they have similar quality, service and ambiance ratings to drive thru users and convenience shoppers, they are less likely to recommend Starbucks as a meeting place and a slightly less likely to return to Starbucks in the future. This is likely explained by their lower perception of available price options. This group likes Starbucks but may be income limited.

**Group 5: Budget students**<br> 
This group of students typically enjoys their Starbucks on location and enjoys the ambiance similarly to other groups. However, they are more critical of the product quality, service and price offerings and spend less money per order. Likely, this is due to having a limiting budget and choosing to be more frugal.

<h2><br></h2>
<h2>Conclusions</h2>
The insight provided by this analysis can be used to improve Starbucks consumer loyalty. We see two groups of customers apparent in the segmentation data: Groups 1-3 are not income limited, and groups 4-5 are income limited. Within groups 1-3, the key driver of customer loyalty appears to be enthusiasm for the experience. This is evident in Group 1, who dine in and spend the most per purchase. This indicates that driving more customers into the store through improved ambiance would improve loyalty in Groups 2-3. 

Improving loyalty in Groups 4-5 would require improving perceptions of product pricing and quality, especially in Group 5. Loyalty in Group 4 may be especially price dependent as this group views Starbucks very positively, except for price options. As these groups are composed of students, adding a student discount may be particularly helpful.  Additional student-focused promotions may be required to improve loyalty in Group 5. 

