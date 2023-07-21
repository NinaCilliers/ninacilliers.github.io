---
layout: page
title: An optimized pipeline for Airbnb price prediction
description: A pipeline is developed to predict Airbnb rental Airbnb prices in Europe. The performance of six data preprocessing strategies was evaluated in nine models against two error metrics. 
img: assets/img/project_previews/airbnb_pipeline.png
importance: 4
category: Scikit-learn
---

<h1>An optimized pipeline for Airbnb price prediction</h1>
<h1> <br> </h1>
<h1>  Introduction </h1>
           
A pipeline is developed to predict the price of Airbnb stays in Europe. An optimized pipeline for Airbnb price forecasting can identify key factors that impact the price of listings and provide insights into market trends and competitors' offerings. By utilizing this pipeline to forecast future revenue based on these insights, hosts can make data-driven decisions that maximize their profits. This streamlined approach saves time, increases accuracy, and delivers the best possible experience to Airbnb guests. 
<br>
<br>
First, we carefully consider the range of prices of interest. Next, we develop several pre-processing strategies that utilize feature re-scaling and feature engineering. Next, we evaluate these strategies in conjunction with common models looking at two common error metrics. Extra trees and random forest regressors both perform, with extra trees performing the best. This model is combined with the optimal pre-processing strategy in a transferrable and concise final pipeline. 
<br>
<br>
<img src="https://news.airbnb.com/wp-content/uploads/sites/4/2020/12/Airbnb-Stay-New-South-Wales.jpg?w=768" width="920px">

<h1> <br></h1>
<h1>  Project Outline </h1>

<h3 href='#selecting'>Selecting price range</h3>
- Raw data is imported from source
- Justification for outlier removal
- Outlier identification

### Exploratory data analysis 
- Heat map 
- Pair plots
- RBF definition
- Visualizing categorical data
- Visualizing rankings

### Data preprocessing
- Encoding categorical data
- Building pipelines for model evaluation

### Price prediction
- Sampling regression models with preprocessing strategies
- Optimizing extra trees regression model

### Conclusion
- The best performing model used an extra trees regressor with 500 estimators to predict the log of the price. 
- This pipeline had mean squared error of 11,788.

<h1> <br> </h1>
<h1> <A id='selecting'>  Selecting price range </A></h1>         
## Importing data


```python
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn import set_config
```


```python
plt.style.use('ggplot')
pd.set_option('display.max_columns', 100)
#set_config(transform_output="pandas") #doesn't work here :(
```


```python
os.chdir('/kaggle/input/airbnb-cleaned-europe-dataset')
df = pd.read_csv('Aemf1.csv')
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
      <th>City</th>
      <th>Price</th>
      <th>Day</th>
      <th>Room Type</th>
      <th>Shared Room</th>
      <th>Private Room</th>
      <th>Person Capacity</th>
      <th>Superhost</th>
      <th>Multiple Rooms</th>
      <th>Business</th>
      <th>Cleanliness Rating</th>
      <th>Guest Satisfaction</th>
      <th>Bedrooms</th>
      <th>City Center (km)</th>
      <th>Metro Distance (km)</th>
      <th>Attraction Index</th>
      <th>Normalised Attraction Index</th>
      <th>Restraunt Index</th>
      <th>Normalised Restraunt Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Amsterdam</td>
      <td>194.033698</td>
      <td>Weekday</td>
      <td>Private room</td>
      <td>False</td>
      <td>True</td>
      <td>2.0</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
      <td>10.0</td>
      <td>93.0</td>
      <td>1</td>
      <td>5.022964</td>
      <td>2.539380</td>
      <td>78.690379</td>
      <td>4.166708</td>
      <td>98.253896</td>
      <td>6.846473</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Amsterdam</td>
      <td>344.245776</td>
      <td>Weekday</td>
      <td>Private room</td>
      <td>False</td>
      <td>True</td>
      <td>4.0</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>8.0</td>
      <td>85.0</td>
      <td>1</td>
      <td>0.488389</td>
      <td>0.239404</td>
      <td>631.176378</td>
      <td>33.421209</td>
      <td>837.280757</td>
      <td>58.342928</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Amsterdam</td>
      <td>264.101422</td>
      <td>Weekday</td>
      <td>Private room</td>
      <td>False</td>
      <td>True</td>
      <td>2.0</td>
      <td>False</td>
      <td>0</td>
      <td>1</td>
      <td>9.0</td>
      <td>87.0</td>
      <td>1</td>
      <td>5.748312</td>
      <td>3.651621</td>
      <td>75.275877</td>
      <td>3.985908</td>
      <td>95.386955</td>
      <td>6.646700</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Amsterdam</td>
      <td>433.529398</td>
      <td>Weekday</td>
      <td>Private room</td>
      <td>False</td>
      <td>True</td>
      <td>4.0</td>
      <td>False</td>
      <td>0</td>
      <td>1</td>
      <td>9.0</td>
      <td>90.0</td>
      <td>2</td>
      <td>0.384862</td>
      <td>0.439876</td>
      <td>493.272534</td>
      <td>26.119108</td>
      <td>875.033098</td>
      <td>60.973565</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Amsterdam</td>
      <td>485.552926</td>
      <td>Weekday</td>
      <td>Private room</td>
      <td>False</td>
      <td>True</td>
      <td>2.0</td>
      <td>True</td>
      <td>0</td>
      <td>0</td>
      <td>10.0</td>
      <td>98.0</td>
      <td>1</td>
      <td>0.544738</td>
      <td>0.318693</td>
      <td>552.830324</td>
      <td>29.272733</td>
      <td>815.305740</td>
      <td>56.811677</td>
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
    </tr>
    <tr>
      <th>41709</th>
      <td>Vienna</td>
      <td>715.938574</td>
      <td>Weekend</td>
      <td>Entire home/apt</td>
      <td>False</td>
      <td>False</td>
      <td>6.0</td>
      <td>False</td>
      <td>0</td>
      <td>1</td>
      <td>10.0</td>
      <td>100.0</td>
      <td>3</td>
      <td>0.530181</td>
      <td>0.135447</td>
      <td>219.402478</td>
      <td>15.712158</td>
      <td>438.756874</td>
      <td>10.604584</td>
    </tr>
    <tr>
      <th>41710</th>
      <td>Vienna</td>
      <td>304.793960</td>
      <td>Weekend</td>
      <td>Entire home/apt</td>
      <td>False</td>
      <td>False</td>
      <td>2.0</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>8.0</td>
      <td>86.0</td>
      <td>1</td>
      <td>0.810205</td>
      <td>0.100839</td>
      <td>204.970121</td>
      <td>14.678608</td>
      <td>342.182813</td>
      <td>8.270427</td>
    </tr>
    <tr>
      <th>41711</th>
      <td>Vienna</td>
      <td>637.168969</td>
      <td>Weekend</td>
      <td>Entire home/apt</td>
      <td>False</td>
      <td>False</td>
      <td>2.0</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>10.0</td>
      <td>93.0</td>
      <td>1</td>
      <td>0.994051</td>
      <td>0.202539</td>
      <td>169.073402</td>
      <td>12.107921</td>
      <td>282.296424</td>
      <td>6.822996</td>
    </tr>
    <tr>
      <th>41712</th>
      <td>Vienna</td>
      <td>301.054157</td>
      <td>Weekend</td>
      <td>Private room</td>
      <td>False</td>
      <td>True</td>
      <td>2.0</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>10.0</td>
      <td>87.0</td>
      <td>1</td>
      <td>3.044100</td>
      <td>0.287435</td>
      <td>109.236574</td>
      <td>7.822803</td>
      <td>158.563398</td>
      <td>3.832416</td>
    </tr>
    <tr>
      <th>41713</th>
      <td>Vienna</td>
      <td>133.230489</td>
      <td>Weekend</td>
      <td>Private room</td>
      <td>False</td>
      <td>True</td>
      <td>4.0</td>
      <td>True</td>
      <td>1</td>
      <td>0</td>
      <td>10.0</td>
      <td>93.0</td>
      <td>1</td>
      <td>1.263932</td>
      <td>0.480903</td>
      <td>150.450381</td>
      <td>10.774264</td>
      <td>225.247293</td>
      <td>5.444140</td>
    </tr>
  </tbody>
</table>
<p>41714 rows × 19 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 41714 entries, 0 to 41713
    Data columns (total 19 columns):
     #   Column                       Non-Null Count  Dtype  
    ---  ------                       --------------  -----  
     0   City                         41714 non-null  object 
     1   Price                        41714 non-null  float64
     2   Day                          41714 non-null  object 
     3   Room Type                    41714 non-null  object 
     4   Shared Room                  41714 non-null  bool   
     5   Private Room                 41714 non-null  bool   
     6   Person Capacity              41714 non-null  float64
     7   Superhost                    41714 non-null  bool   
     8   Multiple Rooms               41714 non-null  int64  
     9   Business                     41714 non-null  int64  
     10  Cleanliness Rating           41714 non-null  float64
     11  Guest Satisfaction           41714 non-null  float64
     12  Bedrooms                     41714 non-null  int64  
     13  City Center (km)             41714 non-null  float64
     14  Metro Distance (km)          41714 non-null  float64
     15  Attraction Index             41714 non-null  float64
     16  Normalised Attraction Index  41714 non-null  float64
     17  Restraunt Index              41714 non-null  float64
     18  Normalised Restraunt Index   41714 non-null  float64
    dtypes: bool(3), float64(10), int64(3), object(3)
    memory usage: 5.2+ MB
    
<h2><br></h2>   
<h2 id='selecting'>Selecting price range</h2>

Developing a general pricing model using extremely expensive stays that are likely outliers may not be appropriate or desirable. The price of extremely expensive stays may not be tied to features described in this dataset, such as luxury amenities or event hosting capabilities. Thus, extreme outliers will be identified and removed from the dataset.


```python
sns.histplot(np.log(df['Price']))
plt.show()
```

![png](/assets/img/airbnb_pipeline/output_8_0.png)
    



```python
df.sort_values('Price', ascending=False).head()
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
      <th>City</th>
      <th>Price</th>
      <th>Day</th>
      <th>Room Type</th>
      <th>Shared Room</th>
      <th>Private Room</th>
      <th>Person Capacity</th>
      <th>Superhost</th>
      <th>Multiple Rooms</th>
      <th>Business</th>
      <th>Cleanliness Rating</th>
      <th>Guest Satisfaction</th>
      <th>Bedrooms</th>
      <th>City Center (km)</th>
      <th>Metro Distance (km)</th>
      <th>Attraction Index</th>
      <th>Normalised Attraction Index</th>
      <th>Restraunt Index</th>
      <th>Normalised Restraunt Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3590</th>
      <td>Athens</td>
      <td>18545.450285</td>
      <td>Weekday</td>
      <td>Entire home/apt</td>
      <td>False</td>
      <td>False</td>
      <td>2.0</td>
      <td>True</td>
      <td>0</td>
      <td>1</td>
      <td>10.0</td>
      <td>100.0</td>
      <td>1</td>
      <td>1.196536</td>
      <td>0.381128</td>
      <td>134.904353</td>
      <td>5.086455</td>
      <td>275.573716</td>
      <td>20.691752</td>
    </tr>
    <tr>
      <th>24810</th>
      <td>Paris</td>
      <td>16445.614689</td>
      <td>Weekday</td>
      <td>Entire home/apt</td>
      <td>False</td>
      <td>False</td>
      <td>2.0</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>9.0</td>
      <td>100.0</td>
      <td>1</td>
      <td>4.602378</td>
      <td>0.118665</td>
      <td>260.896109</td>
      <td>12.700335</td>
      <td>545.826245</td>
      <td>32.072497</td>
    </tr>
    <tr>
      <th>38387</th>
      <td>Vienna</td>
      <td>13664.305916</td>
      <td>Weekday</td>
      <td>Private room</td>
      <td>False</td>
      <td>True</td>
      <td>2.0</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>9.0</td>
      <td>87.0</td>
      <td>1</td>
      <td>2.239501</td>
      <td>0.414395</td>
      <td>128.349070</td>
      <td>9.191812</td>
      <td>201.545043</td>
      <td>4.818080</td>
    </tr>
    <tr>
      <th>40794</th>
      <td>Vienna</td>
      <td>13656.358834</td>
      <td>Weekend</td>
      <td>Private room</td>
      <td>False</td>
      <td>True</td>
      <td>2.0</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>9.0</td>
      <td>87.0</td>
      <td>1</td>
      <td>2.239486</td>
      <td>0.414409</td>
      <td>128.349821</td>
      <td>9.191567</td>
      <td>201.546533</td>
      <td>4.871302</td>
    </tr>
    <tr>
      <th>38222</th>
      <td>Vienna</td>
      <td>12942.991375</td>
      <td>Weekday</td>
      <td>Entire home/apt</td>
      <td>False</td>
      <td>False</td>
      <td>4.0</td>
      <td>False</td>
      <td>0</td>
      <td>1</td>
      <td>7.0</td>
      <td>93.0</td>
      <td>1</td>
      <td>1.497979</td>
      <td>0.396893</td>
      <td>123.776241</td>
      <td>8.864325</td>
      <td>196.019793</td>
      <td>4.685995</td>
    </tr>
  </tbody>
</table>
</div>



To identify outliers, the price distribution from the most expensive case, where the city is Amsterdam and the room capacity is 6, is used to identify outliers for the entire data set.  


```python
from scipy.stats import zscore

df_expensive = df[(df['City']=='Amsterdam')&(df['Person Capacity']==6.0)]

mean_val = np.mean(df_expensive['Price'])
std_dev = np.std(df_expensive['Price'])

threshold = 3

lower_bound = mean_val - threshold * std_dev
upper_bound = mean_val + threshold * std_dev

df_no_outliers = df[df['Price']<upper_bound] #&(z_scores>-4)]

print('With outliers exclusded..')
print(f'New max price: ${upper_bound:.2f}')
print(f'New min price: ${lower_bound:.2f}')

sns.histplot(np.log(df_no_outliers['Price']))
plt.show()
```

    With outliers exclusded..
    New max price: $6315.60
    New min price: $-2974.58
    


![outlier removed price histogram](/assets/img/airbnb_pipeline/output_11_1.png)    
    


<div class="alert alert-block alert-success" style="font-size:18px; font-family:verdana; line-height: 1.7em;">
    📌 &nbsp; Prices above \$6,5000 will not be considered by this model, as these are likely anomalous listings influenced by factors beyond what is represented in our data set. Additional information on luxury amenities or event hosting capabilities is required for a general model to price these properties.  </div>

<h2><br></h2>   
## Data summary


```python
df=df[df['Price']<6500].copy()
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 41703 entries, 0 to 41713
    Data columns (total 19 columns):
     #   Column                       Non-Null Count  Dtype  
    ---  ------                       --------------  -----  
     0   City                         41703 non-null  object 
     1   Price                        41703 non-null  float64
     2   Day                          41703 non-null  object 
     3   Room Type                    41703 non-null  object 
     4   Shared Room                  41703 non-null  bool   
     5   Private Room                 41703 non-null  bool   
     6   Person Capacity              41703 non-null  float64
     7   Superhost                    41703 non-null  bool   
     8   Multiple Rooms               41703 non-null  int64  
     9   Business                     41703 non-null  int64  
     10  Cleanliness Rating           41703 non-null  float64
     11  Guest Satisfaction           41703 non-null  float64
     12  Bedrooms                     41703 non-null  int64  
     13  City Center (km)             41703 non-null  float64
     14  Metro Distance (km)          41703 non-null  float64
     15  Attraction Index             41703 non-null  float64
     16  Normalised Attraction Index  41703 non-null  float64
     17  Restraunt Index              41703 non-null  float64
     18  Normalised Restraunt Index   41703 non-null  float64
    dtypes: bool(3), float64(10), int64(3), object(3)
    memory usage: 5.5+ MB
    

<h1> <br> </h1>
<h1>  EDA </h1>

## Heat map

Correlations between variables are visualized in the heat map below. 
- Cleaner rooms are rated higher. 
- Attractions and restaurants are found in the same locations. 
- City centers are closer to attractions and restaurants.



```python
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
```


![heat map](/assets/img/airbnb_pipeline/output_16_0.png)    
    

<h2><br></h2>   
## Pair plots


```python
#raw data 
sns.pairplot(df[['City Center (km)','Metro Distance (km)','Attraction Index','Restraunt Index','Price']], kind='hist',corner=True);
```


![pair plots](/assets/img/airbnb_pipeline/output_18_0.png)   
    


We see that these features are better represented in log space:


```python
#rescaled data
df_trial = pd.DataFrame()
df_trial['City Center (km)'] = np.log(df['City Center (km)'])
df_trial['Metro Distance (km)'] = np.log(df['Metro Distance (km)'])
df_trial['Attraction Index']=np.log(df['Attraction Index'])
df_trial['Restraunt Index']=np.log(df['Restraunt Index'])
df_trial['Price']=np.log(df['Price'])
sns.pairplot(df_trial, kind='hist',corner=True);
```


![log transformed pair plots](/assets/img/airbnb_pipeline/output_20_0.png)    

<h2><br></h2>   
## RBF Definition
We find a comparable radial basis function to describe features with an apparent radial price distribution in the above pair plots.


```python
#Metro, city center and restraunt index RBFs

from sklearn.metrics.pairwise import rbf_kernel
df['rbf_metro'] = rbf_kernel(df_trial[['Metro Distance (km)']],[[-.5]], gamma=1) 
df['rbf_city'] = rbf_kernel(df_trial[['City Center (km)']],[[.3]], gamma=.5)
df['rbf_res'] = rbf_kernel(df_trial[['Restraunt Index']],[[6.25]], gamma=.5)
```


```python
#visualizing metro rbf function
fig, ax1 = plt.subplots(1)
plt.bar(df_trial['Metro Distance (km)'], df['Price'])
plt.xlabel('Log Metro Distance (km)')
plt.ylabel('Price')
ax2=ax1.twinx()
ax2.scatter(df_trial['Metro Distance (km)'], df['rbf_metro'],color='k',s=.5)
ax2.set_ylim([0,1])
ax2.set_ylabel('Price rbf')
plt.show()
```


![rbf metro](/assets/img/airbnb_pipeline/output_23_0.png)    



```python
#visualizing city rbf function
fig, ax1 = plt.subplots(1)
plt.bar(df_trial['City Center (km)'], df['Price'])
plt.xlabel('Log City Center (km)')
plt.ylabel('Price')
ax2=ax1.twinx()
ax2.scatter(df_trial['City Center (km)'], df['rbf_city'],color='k',s=.5)
ax2.set_ylim([0,1])
ax2.set_ylabel('City rbf')
plt.show()
```


![rbf city center](/assets/img/airbnb_pipeline/output_24_0.png)    
    



```python
#visualizing city rbf function
fig, ax1 = plt.subplots(1)
plt.bar(df_trial['Restraunt Index'], df['Price'])
plt.xlabel('Log Restraunt Index (km)')
plt.ylabel('Price')
ax2=ax1.twinx()
ax2.scatter(df_trial['Restraunt Index'], df['rbf_res'],color='k',s=.5)
ax2.set_ylim([0,1])
ax2.set_ylabel('Restraunt rbf')
plt.show()
```


![rbf restaurant](/assets/img/airbnb_pipeline/output_25_0.png)    
    

<h2><br></h2>   
## Visualizing categorical data


```python
fig, ax = plt.subplots(1)
fig.set_size_inches(8,4)
sns.boxplot(data = df, x='City', y='Price',showfliers = False)
ax.set_ylim([0,1300])
plt.show()
```


![price by city](/assets/img/airbnb_pipeline/output_27_0.png)
    



```python
sns.boxplot(data=df, x='Superhost',y='Price',showfliers=False);
```


![superhost box plot](/assets/img/airbnb_pipeline/output_28_0.png)    
    



```python
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(8,3))
plt.sca(ax1)
sns.boxplot(data=df, x='Room Type',y='Price',showfliers=False)
plt.sca(ax2)
sns.boxplot(data=df, x='Shared Room',y='Price',showfliers=False)
plt.tight_layout()
```


![room type boxplot](/assets/img/airbnb_pipeline/output_29_0.png)    



```python
sns.boxplot(data=df, x='Day',y='Price',showfliers=False);
```


![day boxplot](/assets/img/airbnb_pipeline/output_30_0.png)    
    



```python
sns.boxplot(data = df, x='Person Capacity', y='Price',showfliers=False)
plt.ylim([0,1000])
plt.show()
```


![capacity boxplot](/assets/img/airbnb_pipeline/output_31_0.png)    
    

<h2><br></h2>   
## Visualizing rankings


```python
#sns.regplot(data = df[df["Price"]<2000], x='Cleanliness Rating', y='Price',scatter=True,scatter_kws={'alpha':0.05},line_kws={"color": "black"});
sns.jointplot(x=df['Cleanliness Rating'],y=np.log(df['Price']),kind='reg',scatter_kws={'alpha':0.05},line_kws={"color": "black"})
plt.show()
```


![cleanliness rating](/assets/img/airbnb_pipeline/output_33_0.png)    
    



```python
#sns.regplot(data = df[df['Price']<2000], x='Guest Satisfaction', y='Price',scatter_kws={'alpha':0.05},line_kws={"color": "black"})
sns.jointplot(x=df['Guest Satisfaction'],y=np.log(df['Price']),kind='reg',scatter_kws={'alpha':0.05},line_kws={"color": "black"})
plt.show()
```


![guest satisfaction](/assets/img/airbnb_pipeline/output_34_0.png)    
    


<h1><br></h1>
<h1>  Data preprocessing </h1>

#### Column transformers are developed to evaluate the effect of three different preprocessing strategies on model efficacy. 


```python
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
```


```python
#defining functions for column transformer

cat_encoder = OneHotEncoder()

log_pipeline = make_pipeline(
    FunctionTransformer(np.log), #, inverse_func=np.exp),
    MinMaxScaler())

def day_2_num(X):
    return X=='Weekend'

def day_pipeline():
    return make_pipeline(FunctionTransformer(day_2_num))
        
day_pipe = day_pipeline()
```
<h2><br></h2>   
## 1. Standard preprocessing 
- Normalized attraction index and restaurant index are not used
- Data is scaled using a min-max scaling strategy
- Restaurant index, attraction index, city center and metro index are log-transformed. 
- Room type and city are encoded. 


```python
#defining standard column transformer
preprocessing = ColumnTransformer([
    ('day', day_pipe, ['Day']),
    ('drop','drop',['Normalised Attraction Index','Normalised Restraunt Index','rbf_metro','rbf_city','rbf_res']),
    ('pass', 'passthrough',['Private Room','Shared Room','Superhost','Business','Multiple Rooms']),
    ('maxscale', MinMaxScaler(),['Cleanliness Rating','Bedrooms','Guest Satisfaction']),
    ('log',log_pipeline,['Attraction Index','City Center (km)','Metro Distance (km)','Restraunt Index']),
    ('cat', cat_encoder, ['Room Type','City'])
])
```
<h2><br></h2>   
## 2. Preprocessing using developed RBFs
- Normalized attraction index and restaurant index are not used
- Data is scaled using a min-max scaling strategy
- Attraction index is log-transformed. 
- Restaurant index, metro index, and city center are replaced with the RBF functions developed during EDA.
- Room type and city are encoded. 


```python
#column transformer with rbf functions instead of metro, city, and restraunts
preprocessing_rbf = ColumnTransformer([
    ('day', day_pipe, ['Day']),
    ('drop','drop',['Normalised Attraction Index','Normalised Restraunt Index','Metro Distance (km)','City Center (km)','Restraunt Index']),
    ('pass', 'passthrough',['Private Room','Shared Room','Superhost','Business','Multiple Rooms']),
    ('maxscale', MinMaxScaler(),['Cleanliness Rating','Bedrooms','Guest Satisfaction']),
    ('log',log_pipeline,['Attraction Index']),
    ('pass2', 'passthrough',['rbf_metro','rbf_city','rbf_res']),                 
    ('cat', cat_encoder, ['Room Type','City'])
])
```
<h2><br></h2>   
## 3. Preprocessing using given normalized restaurant and attraction indexes 
- Data is scaled using a min-max scaling strategy
- City center and metro index are log-transformed. 
- Room type and city are encoded. 


```python
#column transformer with given normalized features
preprocessing_norm = ColumnTransformer([
    ('day', day_pipe, ['Day']),
    ('drop','drop',['Attraction Index','Restraunt Index','rbf_metro','rbf_city','rbf_res']),
    ('pass', 'passthrough',['Private Room','Shared Room','Superhost','Business','Multiple Rooms']),
    ('maxscale', MinMaxScaler(),['Cleanliness Rating','Bedrooms','Guest Satisfaction']),
    ('pass2','passthrough',['Normalised Attraction Index']),
    ('log',log_pipeline,['City Center (km)','Metro Distance (km)']),
    ('pass3','passthrough',['Normalised Restraunt Index']),
    ('cat', cat_encoder, ['Room Type','City'])
])
```


```python
#naming output columns
names = pd.Series(['Weekend','Private Room','Shared Room','Superhost','Business','Multiple Rooms','Cleanliness Rating','Bedrooms','Guest Satisfaction','Attraction Index','City Center','Metro Distance','Restraunt Index', 'Private room', 'Entire home/apt', 'Shared room','Amsterdam','Athens','Barcelona','Berlin','Budapest','Lisbon','Paris','Rome','Vienna'])
```


```python
#transforming data
df_processed = pd.DataFrame(preprocessing.fit_transform(df), columns = names)
```


```python
#splitting data into test and train sets 
y = df['Price'].copy()
X = df.drop(columns={'Price'})

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=10)
```

<h1><br></h1>
<h1>  Price prediction </h1>


```python
#importing packages
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.decomposition import PCA 
```
<h2><br></h2>   
## Sampling regression models with preprocessing strategies
Common regression models were evaluated with six variations: 

1. Standard preprocessing
2. Standard preprocessing with price target log-transformed
3. Standard preprocessing with dimensionality reduction via PCA
4. Preprocessing using RBFs
5. Preprocessing using provided normalized restaurant and attraction indexes
5. Combining variations 2-4

To evaluate model performance, mean absolute percentage error (MAPE) and mean squared error (MSE) were both used. MSE measures the average squared difference between the predicted values and the actual values. MAPE measures the average percentage difference between the predicted values and the actual values. Unlike MSE, MAPE is a relative measure and is expressed in percentage terms. 

MAPE is particularly useful when the data contains large values or outliers, as it gives equal weight to all the data points regardless of their scale. This can be important because MSE is sensitive to outliers and large values. When you have large data points, their influence on the model's performance can skew the results of MSE, making it harder to determine the accuracy of the model. By including MAPE in the evaluation, you can better assess how well the model is predicting the target variable, even in the presence of large data points.


```python
#defining mean absolute percentage error 
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```


```python
#selected models
np.random.seed(10)
models = [Ridge(), ElasticNet(), SVR(),KNeighborsRegressor(), RandomForestRegressor(), ExtraTreesRegressor(), AdaBoostRegressor(), GradientBoostingRegressor(), xgb.XGBRegressor()]
model_names = ['Ridge','ElasticNet','SVR','K-nearest neighbors','Random Forest','Extra Trees','Adaptive Boosting','Gradient Boosting','XGBoost']
```


```python
#y
mse = []
mape_err = []

for model in models:
    sample_pipe = make_pipeline(preprocessing, model)
    sample_pipe.fit(X_train, y_train)
    y_pred = sample_pipe.predict(X_test)
    mse.append(mean_squared_error(y_test, y_pred))
    mape_err.append(mape(y_test, y_pred))
```


```python
#log y
scaled_mse = []
scaled_mape = []

for model in models:
    scaled_pipe = make_pipeline(preprocessing, model)
    scaled_pipe.fit(X_train, np.log(y_train))
    y_pred = scaled_pipe.predict(X_test)
    y_pred = np.exp(y_pred)
    scaled_mse.append(mean_squared_error(y_test, y_pred))
    scaled_mape.append(mape(y_test, y_pred))
```


```python
#with dimensionality reduction
pca = PCA(n_components=.95)
pca_mse = []
pca_mape = []

for model in models:
    pca_pipe = make_pipeline(preprocessing, pca, model)
    pca_pipe.fit(X_train, y_train)
    y_pred = pca_pipe.predict(X_test)
    pca_mse.append(mean_squared_error(y_test, y_pred))
    pca_mape.append(mape(y_test, y_pred))
```


```python
#with rbf 
rbf_mse = []
rbf_mape = []

for model in models:
    rbf_pipe = make_pipeline(preprocessing_rbf, model)
    rbf_pipe.fit(X_train, y_train)
    y_pred = rbf_pipe.predict(X_test)
    rbf_mse.append(mean_squared_error(y_test, y_pred))
    rbf_mape.append(mape(y_test, y_pred))
```


```python
#with provided normalized features
norm_mse = []
norm_mape = []

for model in models:
    norm_pipe = make_pipeline(preprocessing_norm, model)
    norm_pipe.fit(X_train, y_train)
    y_pred = norm_pipe.predict(X_test)
    norm_mse.append(mean_squared_error(y_test, y_pred))
    norm_mape.append(mape(y_test, y_pred))
```


```python
#combination 
combo_mse = []
combo_mape = []

for model in models:
    combo_pipe = make_pipeline(preprocessing_rbf, pca, model)
    combo_pipe.fit(X_train, np.log(y_train))
    y_pred = combo_pipe.predict(X_test)
    y_pred = np.exp(y_pred)
    combo_mse.append(mean_squared_error(y_test, y_pred))
    combo_mape.append(mape(y_test, y_pred))
```


```python
#error compilation
mse_results = pd.DataFrame([mse,scaled_mse,pca_mse, rbf_mse, norm_mse, combo_mse], index = ['Unscaled','Log price','Reduced Dimensions','RBF Features','Normalized Features','Combination'], columns=model_names).T
mape_results = pd.DataFrame([mape_err,scaled_mape,pca_mape, rbf_mape, norm_mape, combo_mape], index = ['Unscaled','Log price','Reduced Dimensions','RBF Features','Normalized Features','Combination'], columns=model_names).T
```


```python
#visualing results
mse_results.plot(kind='barh')
plt.title('Regressor MSE')
plt.xlim([0,100000])
plt.legend(loc=4, facecolor='white')

mape_results.plot(kind='barh')
plt.title('Regressor MAPE')
plt.xlim([0,100])
plt.legend(loc=4, facecolor='white')
plt.show()
```


![MSE performance](/assets/img/airbnb_pipeline/output_60_0.png)    
    

![MAPE performance](/assets/img/airbnb_pipeline/output_60_1.png)   
    



```python
mse_results.style.format('{:.0f}')
```




<style type="text/css">
</style>
<table id="T_81bd0_">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th class="col_heading level0 col0" >Unscaled</th>
      <th class="col_heading level0 col1" >Log price</th>
      <th class="col_heading level0 col2" >Reduced Dimensions</th>
      <th class="col_heading level0 col3" >RBF Features</th>
      <th class="col_heading level0 col4" >Normalized Features</th>
      <th class="col_heading level0 col5" >Combination</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_81bd0_level0_row0" class="row_heading level0 row0" >Ridge</th>
      <td id="T_81bd0_row0_col0" class="data row0 col0" >27182</td>
      <td id="T_81bd0_row0_col1" class="data row0 col1" >25203</td>
      <td id="T_81bd0_row0_col2" class="data row0 col2" >37715</td>
      <td id="T_81bd0_row0_col3" class="data row0 col3" >27163</td>
      <td id="T_81bd0_row0_col4" class="data row0 col4" >26996</td>
      <td id="T_81bd0_row0_col5" class="data row0 col5" >32218</td>
    </tr>
    <tr>
      <th id="T_81bd0_level0_row1" class="row_heading level0 row1" >ElasticNet</th>
      <td id="T_81bd0_row1_col0" class="data row1 col0" >41868</td>
      <td id="T_81bd0_row1_col1" class="data row1 col1" >49133</td>
      <td id="T_81bd0_row1_col2" class="data row1 col2" >42977</td>
      <td id="T_81bd0_row1_col3" class="data row1 col3" >41882</td>
      <td id="T_81bd0_row1_col4" class="data row1 col4" >38034</td>
      <td id="T_81bd0_row1_col5" class="data row1 col5" >49133</td>
    </tr>
    <tr>
      <th id="T_81bd0_level0_row2" class="row_heading level0 row2" >SVR</th>
      <td id="T_81bd0_row2_col0" class="data row2 col0" >33970</td>
      <td id="T_81bd0_row2_col1" class="data row2 col1" >22371</td>
      <td id="T_81bd0_row2_col2" class="data row2 col2" >37879</td>
      <td id="T_81bd0_row2_col3" class="data row2 col3" >34282</td>
      <td id="T_81bd0_row2_col4" class="data row2 col4" >42966</td>
      <td id="T_81bd0_row2_col5" class="data row2 col5" >29286</td>
    </tr>
    <tr>
      <th id="T_81bd0_level0_row3" class="row_heading level0 row3" >K-nearest neighbors</th>
      <td id="T_81bd0_row3_col0" class="data row3 col0" >25041</td>
      <td id="T_81bd0_row3_col1" class="data row3 col1" >24365</td>
      <td id="T_81bd0_row3_col2" class="data row3 col2" >29786</td>
      <td id="T_81bd0_row3_col3" class="data row3 col3" >26185</td>
      <td id="T_81bd0_row3_col4" class="data row3 col4" >33192</td>
      <td id="T_81bd0_row3_col5" class="data row3 col5" >29520</td>
    </tr>
    <tr>
      <th id="T_81bd0_level0_row4" class="row_heading level0 row4" >Random Forest</th>
      <td id="T_81bd0_row4_col0" class="data row4 col0" >13782</td>
      <td id="T_81bd0_row4_col1" class="data row4 col1" >14281</td>
      <td id="T_81bd0_row4_col2" class="data row4 col2" >25219</td>
      <td id="T_81bd0_row4_col3" class="data row4 col3" >13657</td>
      <td id="T_81bd0_row4_col4" class="data row4 col4" >13892</td>
      <td id="T_81bd0_row4_col5" class="data row4 col5" >25716</td>
    </tr>
    <tr>
      <th id="T_81bd0_level0_row5" class="row_heading level0 row5" >Extra Trees</th>
      <td id="T_81bd0_row5_col0" class="data row5 col0" >11918</td>
      <td id="T_81bd0_row5_col1" class="data row5 col1" >11824</td>
      <td id="T_81bd0_row5_col2" class="data row5 col2" >25528</td>
      <td id="T_81bd0_row5_col3" class="data row5 col3" >12174</td>
      <td id="T_81bd0_row5_col4" class="data row5 col4" >13825</td>
      <td id="T_81bd0_row5_col5" class="data row5 col5" >25473</td>
    </tr>
    <tr>
      <th id="T_81bd0_level0_row6" class="row_heading level0 row6" >Adaptive Boosting</th>
      <td id="T_81bd0_row6_col0" class="data row6 col0" >73914</td>
      <td id="T_81bd0_row6_col1" class="data row6 col1" >33217</td>
      <td id="T_81bd0_row6_col2" class="data row6 col2" >44755</td>
      <td id="T_81bd0_row6_col3" class="data row6 col3" >186115</td>
      <td id="T_81bd0_row6_col4" class="data row6 col4" >176752</td>
      <td id="T_81bd0_row6_col5" class="data row6 col5" >40706</td>
    </tr>
    <tr>
      <th id="T_81bd0_level0_row7" class="row_heading level0 row7" >Gradient Boosting</th>
      <td id="T_81bd0_row7_col0" class="data row7 col0" >20228</td>
      <td id="T_81bd0_row7_col1" class="data row7 col1" >22547</td>
      <td id="T_81bd0_row7_col2" class="data row7 col2" >29095</td>
      <td id="T_81bd0_row7_col3" class="data row7 col3" >20371</td>
      <td id="T_81bd0_row7_col4" class="data row7 col4" >20465</td>
      <td id="T_81bd0_row7_col5" class="data row7 col5" >30652</td>
    </tr>
    <tr>
      <th id="T_81bd0_level0_row8" class="row_heading level0 row8" >XGBoost</th>
      <td id="T_81bd0_row8_col0" class="data row8 col0" >15636</td>
      <td id="T_81bd0_row8_col1" class="data row8 col1" >17635</td>
      <td id="T_81bd0_row8_col2" class="data row8 col2" >27276</td>
      <td id="T_81bd0_row8_col3" class="data row8 col3" >14857</td>
      <td id="T_81bd0_row8_col4" class="data row8 col4" >16600</td>
      <td id="T_81bd0_row8_col5" class="data row8 col5" >27639</td>
    </tr>
  </tbody>
</table>





```python
mape_results.style.format('{:.1f}')
```




<style type="text/css">
</style>
<table id="T_cf2f2_">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th class="col_heading level0 col0" >Unscaled</th>
      <th class="col_heading level0 col1" >Log price</th>
      <th class="col_heading level0 col2" >Reduced Dimensions</th>
      <th class="col_heading level0 col3" >RBF Features</th>
      <th class="col_heading level0 col4" >Normalized Features</th>
      <th class="col_heading level0 col5" >Combination</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_cf2f2_level0_row0" class="row_heading level0 row0" >Ridge</th>
      <td id="T_cf2f2_row0_col0" class="data row0 col0" >35.5</td>
      <td id="T_cf2f2_row0_col1" class="data row0 col1" >25.4</td>
      <td id="T_cf2f2_row0_col2" class="data row0 col2" >41.6</td>
      <td id="T_cf2f2_row0_col3" class="data row0 col3" >35.4</td>
      <td id="T_cf2f2_row0_col4" class="data row0 col4" >34.4</td>
      <td id="T_cf2f2_row0_col5" class="data row0 col5" >30.2</td>
    </tr>
    <tr>
      <th id="T_cf2f2_level0_row1" class="row_heading level0 row1" >ElasticNet</th>
      <td id="T_cf2f2_row1_col0" class="data row1 col0" >52.2</td>
      <td id="T_cf2f2_row1_col1" class="data row1 col1" >46.9</td>
      <td id="T_cf2f2_row1_col2" class="data row1 col2" >53.3</td>
      <td id="T_cf2f2_row1_col3" class="data row1 col3" >52.2</td>
      <td id="T_cf2f2_row1_col4" class="data row1 col4" >43.3</td>
      <td id="T_cf2f2_row1_col5" class="data row1 col5" >46.9</td>
    </tr>
    <tr>
      <th id="T_cf2f2_level0_row2" class="row_heading level0 row2" >SVR</th>
      <td id="T_cf2f2_row2_col0" class="data row2 col0" >25.7</td>
      <td id="T_cf2f2_row2_col1" class="data row2 col1" >22.7</td>
      <td id="T_cf2f2_row2_col2" class="data row2 col2" >29.2</td>
      <td id="T_cf2f2_row2_col3" class="data row2 col3" >25.9</td>
      <td id="T_cf2f2_row2_col4" class="data row2 col4" >35.9</td>
      <td id="T_cf2f2_row2_col5" class="data row2 col5" >25.2</td>
    </tr>
    <tr>
      <th id="T_cf2f2_level0_row3" class="row_heading level0 row3" >K-nearest neighbors</th>
      <td id="T_cf2f2_row3_col0" class="data row3 col0" >27.2</td>
      <td id="T_cf2f2_row3_col1" class="data row3 col1" >24.8</td>
      <td id="T_cf2f2_row3_col2" class="data row3 col2" >30.3</td>
      <td id="T_cf2f2_row3_col3" class="data row3 col3" >27.9</td>
      <td id="T_cf2f2_row3_col4" class="data row3 col4" >32.5</td>
      <td id="T_cf2f2_row3_col5" class="data row3 col5" >28.6</td>
    </tr>
    <tr>
      <th id="T_cf2f2_level0_row4" class="row_heading level0 row4" >Random Forest</th>
      <td id="T_cf2f2_row4_col0" class="data row4 col0" >18.4</td>
      <td id="T_cf2f2_row4_col1" class="data row4 col1" >16.5</td>
      <td id="T_cf2f2_row4_col2" class="data row4 col2" >28.1</td>
      <td id="T_cf2f2_row4_col3" class="data row4 col3" >18.6</td>
      <td id="T_cf2f2_row4_col4" class="data row4 col4" >20.0</td>
      <td id="T_cf2f2_row4_col5" class="data row4 col5" >24.8</td>
    </tr>
    <tr>
      <th id="T_cf2f2_level0_row5" class="row_heading level0 row5" >Extra Trees</th>
      <td id="T_cf2f2_row5_col0" class="data row5 col0" >17.0</td>
      <td id="T_cf2f2_row5_col1" class="data row5 col1" >15.2</td>
      <td id="T_cf2f2_row5_col2" class="data row5 col2" >28.2</td>
      <td id="T_cf2f2_row5_col3" class="data row5 col3" >17.1</td>
      <td id="T_cf2f2_row5_col4" class="data row5 col4" >19.3</td>
      <td id="T_cf2f2_row5_col5" class="data row5 col5" >24.9</td>
    </tr>
    <tr>
      <th id="T_cf2f2_level0_row6" class="row_heading level0 row6" >Adaptive Boosting</th>
      <td id="T_cf2f2_row6_col0" class="data row6 col0" >121.6</td>
      <td id="T_cf2f2_row6_col1" class="data row6 col1" >56.8</td>
      <td id="T_cf2f2_row6_col2" class="data row6 col2" >70.6</td>
      <td id="T_cf2f2_row6_col3" class="data row6 col3" >205.0</td>
      <td id="T_cf2f2_row6_col4" class="data row6 col4" >164.5</td>
      <td id="T_cf2f2_row6_col5" class="data row6 col5" >56.9</td>
    </tr>
    <tr>
      <th id="T_cf2f2_level0_row7" class="row_heading level0 row7" >Gradient Boosting</th>
      <td id="T_cf2f2_row7_col0" class="data row7 col0" >27.3</td>
      <td id="T_cf2f2_row7_col1" class="data row7 col1" >23.8</td>
      <td id="T_cf2f2_row7_col2" class="data row7 col2" >32.8</td>
      <td id="T_cf2f2_row7_col3" class="data row7 col3" >27.5</td>
      <td id="T_cf2f2_row7_col4" class="data row7 col4" >27.0</td>
      <td id="T_cf2f2_row7_col5" class="data row7 col5" >28.4</td>
    </tr>
    <tr>
      <th id="T_cf2f2_level0_row8" class="row_heading level0 row8" >XGBoost</th>
      <td id="T_cf2f2_row8_col0" class="data row8 col0" >24.2</td>
      <td id="T_cf2f2_row8_col1" class="data row8 col1" >21.2</td>
      <td id="T_cf2f2_row8_col2" class="data row8 col2" >29.7</td>
      <td id="T_cf2f2_row8_col3" class="data row8 col3" >24.4</td>
      <td id="T_cf2f2_row8_col4" class="data row8 col4" >24.4</td>
      <td id="T_cf2f2_row8_col5" class="data row8 col5" >26.3</td>
    </tr>
  </tbody>
</table>




<div class="alert alert-block alert-success" style="font-size:18px; font-family:verdana; line-height: 1.7em;">
    📌 &nbsp; The top performing regressor was the extra trees regressor with the log-transformed price.  </div>

<h2><br></h2>   
## Optimizing extra trees regression model

#### General dependence of MSE on number of estimators


```python
from sklearn.model_selection import GridSearchCV 
```


```python
mse_n_est = []

n_estimators=np.arange(10,1001,10)
for n in n_estimators:
    tree_opt = make_pipeline(preprocessing, ExtraTreesRegressor(n_jobs=-1,random_state=10, n_estimators=n))
    tree_opt.fit(X_train,np.log(y_train))
    y_pred = tree_opt.predict(X_test)
    y_pred = np.exp(y_pred)
    mse_n_est.append(mean_squared_error(y_test, y_pred))

plt.plot(n_estimators,mse_n_est,'-')
plt.xlabel('n_estimators')
plt.ylabel('mean squared error')
plt.show()
```


![extra tree regressor tree number optimization](/assets/img/airbnb_pipeline/output_67_0.png)    
    



```python
print(f'Mean squared error at n_estimators = 500: {mse_n_est[-1]:.0f}.')
```

    Mean squared error at n_estimators = 500: 11796.
    

<div class="alert alert-block alert-success" style="font-size:18px; font-family:verdana; line-height: 1.7em;">
    📌 &nbsp;  We will use 500 trees to optimize other model parameters, as this number minimizes the mean squared error without overfitting.  </div>

#### GridSearchCV


```python
#grid search 

tree_pipe = make_pipeline(preprocessing, ExtraTreesRegressor(n_jobs=-1,random_state=10))

n_estimators = [500]
max_features = ['sqrt'] #['sqrt','log2'] sqrt was shown to consistantly outperform log 
max_depth = np.arange(10,100,5)


tree_params = {
    'extratreesregressor__n_estimators':n_estimators,
    'extratreesregressor__max_features':max_features,
    'extratreesregressor__max_depth':max_depth
}

tree_search = GridSearchCV(tree_pipe,tree_params,scoring='neg_root_mean_squared_error',cv=5,n_jobs=-1)
tree_search.fit(X_train,np.log(y_train))
tree_search.best_score_,tree_search.best_params_
```

    /opt/conda/lib/python3.7/site-packages/joblib/externals/loky/process_executor.py:703: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.
      "timeout or by a memory leak.", UserWarning
    




    (-0.27009412940530914,
     {'extratreesregressor__max_depth': 30,
      'extratreesregressor__max_features': 'sqrt',
      'extratreesregressor__n_estimators': 500})




```python
y_pred = tree_search.predict(X_test)
mean_squared_error(y_test, np.exp(y_pred))
```




    16595.141980500048



Constraining the max_depth appears to weaken the predictive power of the extra trees regressor in this case. Therefore, only n_estimators will be adjusted as a hyperparameter and the lowest MSE is 11,788. 

<h1><br></h1>
<h1>  Conclusion </h1>

The analysis done in this project enables us to build an optimized pipeline for predicting Airbnb prices. The steps in this pipeline are summarized below:

1. Attraction index, restaurant index, city center and metro index are log-transformed. 
2. Room type and city are encoded. 
3. Data is scaled using a min-max scaling strategy. 
4. Housing prices are log-transformed. 
5. Log-price is predicted using a trained and optimized random forest regressor 
6. Price is transformed back to linear space. 

<br>
The top performing model on this dataset is defined for clarity below:



```python
set_config(display='diagram')
```


```python
transformer = FunctionTransformer(func=np.log, inverse_func=np.exp)
regressor = TransformedTargetRegressor(regressor=ExtraTreesRegressor(n_jobs=-1,random_state=10, n_estimators=500),
                                      transformer=transformer)

top_pipeline = make_pipeline(
    preprocessing, 
    regressor)

top_pipeline
```




<style>#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 {color: black;background-color: white;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 pre{padding: 0;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 div.sk-toggleable {background-color: white;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 div.sk-item {z-index: 1;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 div.sk-parallel::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 div.sk-parallel-item:only-child::after {width: 0;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;position: relative;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-24dc161d-9ff3-4410-baaa-6cc977b7c5c3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;columntransformer&#x27;,
                 ColumnTransformer(transformers=[(&#x27;day&#x27;,
                                                  Pipeline(steps=[(&#x27;functiontransformer&#x27;,
                                                                   FunctionTransformer(func=&lt;function day_2_num at 0x71e88f7d0440&gt;))]),
                                                  [&#x27;Day&#x27;]),
                                                 (&#x27;drop&#x27;, &#x27;drop&#x27;,
                                                  [&#x27;Normalised Attraction &#x27;
                                                   &#x27;Index&#x27;,
                                                   &#x27;Normalised Restraunt Index&#x27;,
                                                   &#x27;rbf_metro&#x27;, &#x27;rbf_city&#x27;,
                                                   &#x27;rbf_res&#x27;]),
                                                 (&#x27;pass&#x27;, &#x27;passthrough&#x27;,
                                                  [&#x27;Private Room&#x27;,
                                                   &#x27;Shared Room&#x27;, &#x27;Su...
                                                                   MinMaxScaler())]),
                                                  [&#x27;Attraction Index&#x27;,
                                                   &#x27;City Center (km)&#x27;,
                                                   &#x27;Metro Distance (km)&#x27;,
                                                   &#x27;Restraunt Index&#x27;]),
                                                 (&#x27;cat&#x27;, OneHotEncoder(),
                                                  [&#x27;Room Type&#x27;, &#x27;City&#x27;])])),
                (&#x27;transformedtargetregressor&#x27;,
                 TransformedTargetRegressor(regressor=ExtraTreesRegressor(n_estimators=500,
                                                                          n_jobs=-1,
                                                                          random_state=10),
                                            transformer=FunctionTransformer(func=&lt;ufunc &#x27;log&#x27;&gt;,
                                                                            inverse_func=&lt;ufunc &#x27;exp&#x27;&gt;)))])</pre><b>Please rerun this cell to show the HTML repr or trust the notebook.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="5565bc88-e9b9-44c7-978f-49a485e86364" type="checkbox" ><label for="5565bc88-e9b9-44c7-978f-49a485e86364" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;columntransformer&#x27;,
                 ColumnTransformer(transformers=[(&#x27;day&#x27;,
                                                  Pipeline(steps=[(&#x27;functiontransformer&#x27;,
                                                                   FunctionTransformer(func=&lt;function day_2_num at 0x71e88f7d0440&gt;))]),
                                                  [&#x27;Day&#x27;]),
                                                 (&#x27;drop&#x27;, &#x27;drop&#x27;,
                                                  [&#x27;Normalised Attraction &#x27;
                                                   &#x27;Index&#x27;,
                                                   &#x27;Normalised Restraunt Index&#x27;,
                                                   &#x27;rbf_metro&#x27;, &#x27;rbf_city&#x27;,
                                                   &#x27;rbf_res&#x27;]),
                                                 (&#x27;pass&#x27;, &#x27;passthrough&#x27;,
                                                  [&#x27;Private Room&#x27;,
                                                   &#x27;Shared Room&#x27;, &#x27;Su...
                                                                   MinMaxScaler())]),
                                                  [&#x27;Attraction Index&#x27;,
                                                   &#x27;City Center (km)&#x27;,
                                                   &#x27;Metro Distance (km)&#x27;,
                                                   &#x27;Restraunt Index&#x27;]),
                                                 (&#x27;cat&#x27;, OneHotEncoder(),
                                                  [&#x27;Room Type&#x27;, &#x27;City&#x27;])])),
                (&#x27;transformedtargetregressor&#x27;,
                 TransformedTargetRegressor(regressor=ExtraTreesRegressor(n_estimators=500,
                                                                          n_jobs=-1,
                                                                          random_state=10),
                                            transformer=FunctionTransformer(func=&lt;ufunc &#x27;log&#x27;&gt;,
                                                                            inverse_func=&lt;ufunc &#x27;exp&#x27;&gt;)))])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="3c89f2db-f3b0-48ab-90c8-602b92fcae26" type="checkbox" ><label for="3c89f2db-f3b0-48ab-90c8-602b92fcae26" class="sk-toggleable__label sk-toggleable__label-arrow">columntransformer: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(transformers=[(&#x27;day&#x27;,
                                 Pipeline(steps=[(&#x27;functiontransformer&#x27;,
                                                  FunctionTransformer(func=&lt;function day_2_num at 0x71e88f7d0440&gt;))]),
                                 [&#x27;Day&#x27;]),
                                (&#x27;drop&#x27;, &#x27;drop&#x27;,
                                 [&#x27;Normalised Attraction Index&#x27;,
                                  &#x27;Normalised Restraunt Index&#x27;, &#x27;rbf_metro&#x27;,
                                  &#x27;rbf_city&#x27;, &#x27;rbf_res&#x27;]),
                                (&#x27;pass&#x27;, &#x27;passthrough&#x27;,
                                 [&#x27;Private Room&#x27;, &#x27;Shared Room&#x27;, &#x27;Superhost&#x27;,
                                  &#x27;Business&#x27;, &#x27;Multiple Rooms&#x27;]),
                                (&#x27;maxscale&#x27;, MinMaxScaler(),
                                 [&#x27;Cleanliness Rating&#x27;, &#x27;Bedrooms&#x27;,
                                  &#x27;Guest Satisfaction&#x27;]),
                                (&#x27;log&#x27;,
                                 Pipeline(steps=[(&#x27;functiontransformer&#x27;,
                                                  FunctionTransformer(func=&lt;ufunc &#x27;log&#x27;&gt;)),
                                                 (&#x27;minmaxscaler&#x27;,
                                                  MinMaxScaler())]),
                                 [&#x27;Attraction Index&#x27;, &#x27;City Center (km)&#x27;,
                                  &#x27;Metro Distance (km)&#x27;, &#x27;Restraunt Index&#x27;]),
                                (&#x27;cat&#x27;, OneHotEncoder(),
                                 [&#x27;Room Type&#x27;, &#x27;City&#x27;])])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="8dfbdfcc-4c8b-45a7-8f23-4110fd67eb84" type="checkbox" ><label for="8dfbdfcc-4c8b-45a7-8f23-4110fd67eb84" class="sk-toggleable__label sk-toggleable__label-arrow">day</label><div class="sk-toggleable__content"><pre>[&#x27;Day&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="1bb7d3bb-356f-4196-999f-e54e6ae7847c" type="checkbox" ><label for="1bb7d3bb-356f-4196-999f-e54e6ae7847c" class="sk-toggleable__label sk-toggleable__label-arrow">FunctionTransformer</label><div class="sk-toggleable__content"><pre>FunctionTransformer(func=&lt;function day_2_num at 0x71e88f7d0440&gt;)</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="4fdd4622-e3eb-4298-8ba6-63d39ab32a38" type="checkbox" ><label for="4fdd4622-e3eb-4298-8ba6-63d39ab32a38" class="sk-toggleable__label sk-toggleable__label-arrow">drop</label><div class="sk-toggleable__content"><pre>[&#x27;Normalised Attraction Index&#x27;, &#x27;Normalised Restraunt Index&#x27;, &#x27;rbf_metro&#x27;, &#x27;rbf_city&#x27;, &#x27;rbf_res&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="8b1c406e-d27c-4a9d-b1e2-0270bac5c14a" type="checkbox" ><label for="8b1c406e-d27c-4a9d-b1e2-0270bac5c14a" class="sk-toggleable__label sk-toggleable__label-arrow">drop</label><div class="sk-toggleable__content"><pre>drop</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="066689e8-1740-4c48-9dff-c9b46eb0f5e7" type="checkbox" ><label for="066689e8-1740-4c48-9dff-c9b46eb0f5e7" class="sk-toggleable__label sk-toggleable__label-arrow">pass</label><div class="sk-toggleable__content"><pre>[&#x27;Private Room&#x27;, &#x27;Shared Room&#x27;, &#x27;Superhost&#x27;, &#x27;Business&#x27;, &#x27;Multiple Rooms&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="daf391ba-38ac-4669-9127-66dfe4787e44" type="checkbox" ><label for="daf391ba-38ac-4669-9127-66dfe4787e44" class="sk-toggleable__label sk-toggleable__label-arrow">passthrough</label><div class="sk-toggleable__content"><pre>passthrough</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="cf136e71-56f5-488d-a105-371c8da25e0c" type="checkbox" ><label for="cf136e71-56f5-488d-a105-371c8da25e0c" class="sk-toggleable__label sk-toggleable__label-arrow">maxscale</label><div class="sk-toggleable__content"><pre>[&#x27;Cleanliness Rating&#x27;, &#x27;Bedrooms&#x27;, &#x27;Guest Satisfaction&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="2d27d587-0c58-431d-9300-3dfc193aa07f" type="checkbox" ><label for="2d27d587-0c58-431d-9300-3dfc193aa07f" class="sk-toggleable__label sk-toggleable__label-arrow">MinMaxScaler</label><div class="sk-toggleable__content"><pre>MinMaxScaler()</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="e57ce64a-095e-46c6-ae8b-b3334ffb29e3" type="checkbox" ><label for="e57ce64a-095e-46c6-ae8b-b3334ffb29e3" class="sk-toggleable__label sk-toggleable__label-arrow">log</label><div class="sk-toggleable__content"><pre>[&#x27;Attraction Index&#x27;, &#x27;City Center (km)&#x27;, &#x27;Metro Distance (km)&#x27;, &#x27;Restraunt Index&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="0f5c483b-d6b5-49c0-962d-c03b96fe3769" type="checkbox" ><label for="0f5c483b-d6b5-49c0-962d-c03b96fe3769" class="sk-toggleable__label sk-toggleable__label-arrow">FunctionTransformer</label><div class="sk-toggleable__content"><pre>FunctionTransformer(func=&lt;ufunc &#x27;log&#x27;&gt;)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="0910d8eb-f422-4fa1-bd78-7df09d6374e8" type="checkbox" ><label for="0910d8eb-f422-4fa1-bd78-7df09d6374e8" class="sk-toggleable__label sk-toggleable__label-arrow">MinMaxScaler</label><div class="sk-toggleable__content"><pre>MinMaxScaler()</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="fb766bd5-62de-46bf-baaf-dc7d1dd844dd" type="checkbox" ><label for="fb766bd5-62de-46bf-baaf-dc7d1dd844dd" class="sk-toggleable__label sk-toggleable__label-arrow">cat</label><div class="sk-toggleable__content"><pre>[&#x27;Room Type&#x27;, &#x27;City&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="48c8ef26-5d65-4c79-a716-fdaf38669365" type="checkbox" ><label for="48c8ef26-5d65-4c79-a716-fdaf38669365" class="sk-toggleable__label sk-toggleable__label-arrow">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder()</pre></div></div></div></div></div></div></div></div><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="23f86c80-c408-4209-a9e0-4bd71a200a1b" type="checkbox" ><label for="23f86c80-c408-4209-a9e0-4bd71a200a1b" class="sk-toggleable__label sk-toggleable__label-arrow">transformedtargetregressor: TransformedTargetRegressor</label><div class="sk-toggleable__content"><pre>TransformedTargetRegressor(regressor=ExtraTreesRegressor(n_estimators=500,
                                                         n_jobs=-1,
                                                         random_state=10),
                           transformer=FunctionTransformer(func=&lt;ufunc &#x27;log&#x27;&gt;,
                                                           inverse_func=&lt;ufunc &#x27;exp&#x27;&gt;))</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="0c7c49a6-28ec-42d3-a69d-15a38700d1ec" type="checkbox" ><label for="0c7c49a6-28ec-42d3-a69d-15a38700d1ec" class="sk-toggleable__label sk-toggleable__label-arrow">ExtraTreesRegressor</label><div class="sk-toggleable__content"><pre>ExtraTreesRegressor(n_estimators=500, n_jobs=-1, random_state=10)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="82524cbd-6a16-4990-8b68-84008f5d08a0" type="checkbox" ><label for="82524cbd-6a16-4990-8b68-84008f5d08a0" class="sk-toggleable__label sk-toggleable__label-arrow">FunctionTransformer</label><div class="sk-toggleable__content"><pre>FunctionTransformer(func=&lt;ufunc &#x27;log&#x27;&gt;, inverse_func=&lt;ufunc &#x27;exp&#x27;&gt;)</pre></div></div></div></div></div></div></div></div></div></div></div></div>



And the mean squared error from this pipeline is 11,788.


```python
top_pipeline.fit(X_train, y_train)
y_pred = top_pipeline.predict(X_test)
mean_squared_error(y_test, y_pred)
```




    11787.637281203217


