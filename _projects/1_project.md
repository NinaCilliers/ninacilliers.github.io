---
layout: page
title: Project 1
description: a project with a background image
img: assets/img/3.jpg
importance: 2
category: work
---

# Introduction

CrossFit is a high-intensity fitness program that combines elements of weightlifting, cardio, and gymnastics. It aims to improve overall physical fitness by incorporating constantly varied functional movements performed at a high intensity. At the pinnacle of CrossFit is the CrossFit Games, an annual competition that showcases the world's fittest athletes. The CrossFit Games serve as a platform for elite athletes to test their skills and compete in a wide range of demanding workouts, challenging their strength, speed, power, and mental resilience. In this analysis, we will delve into the performance of CrossFit athletes, examining key factors that contribute to their success in this highly demanding and competitive sport.  

In this comprehensive two-part project, we first clean highly disorganized survey data profiling athletes competing in the CrossFit games. This data encompasses their age, height, weight, training habits (including rest days and sessions per day), geographic location, eating habits (such as measuring food, adhering to paleo or quality eating practices), and performance data. Features are engineered for a more insightful analysis of athlete performance, enabling us to uncover valuable trends and patterns. Notably, strong relationships were found between performance and athlete age, body composition and gender and athlete performance.

In the second part of this project, we use this data to predict athlete performance. Three models are made to this end using common regression models: 

 - Random forest 
 - XGBoost
 - Dense neural network

 XG Boost was found to outperform other models by ~3%. The most important features were extracted using a custom feature permutation algorithm, and gender, age and BMI were found to be the most important considerations, in line with the findings in our data exploration phase. The dense neural network suffered from a higher bias and was not optimal for our dataset. Finally, the error metric from these models is contextualized in terms of my own ability to predict performance. Insight is provided for both athletes and future survey construction. 


# Project Outline 

- **Data Preparation**
   - Data summary
   - Outlier removal
   - Cleaning survey data
   

- **Feature Engineering**
   - Normalizing lifts 
   - Calculating body mass index (BMI)
   
   
- **EDA** 
   - Lifts
   - Region 
   - BMI
   - Age
   - Athlete lifestyle
   - Has CrossFit been life changing?
   - Correlation matrix
   
   
- **Predicting Normalized Total Lift**
  - Random forest regression model
  - XGBoost
  - Feature Importance
  - Dense neural network
  - Benchmarking model performance
  
  
- **Conclusions and recommendations**
  - Advice for athletes and coaches
  - Survey suggestions


# Data Preparation


```python
import pandas as pd
import missingno as msno
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
```

### Data summary


```python
df = pd.read_csv('/kaggle/input/crossfit-athletes/athletes.csv')
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 423006 entries, 0 to 423005
    Data columns (total 27 columns):
     #   Column      Non-Null Count   Dtype  
    ---  ------      --------------   -----  
     0   athlete_id  423003 non-null  float64
     1   name        331110 non-null  object 
     2   region      251262 non-null  object 
     3   team        155160 non-null  object 
     4   affiliate   241916 non-null  object 
     5   gender      331110 non-null  object 
     6   age         331110 non-null  float64
     7   height      159869 non-null  float64
     8   weight      229890 non-null  float64
     9   fran        55426 non-null   float64
     10  helen       30279 non-null   float64
     11  grace       40745 non-null   float64
     12  filthy50    19359 non-null   float64
     13  fgonebad    29738 non-null   float64
     14  run400      22246 non-null   float64
     15  run5k       36097 non-null   float64
     16  candj       104435 non-null  float64
     17  snatch      97280 non-null   float64
     18  deadlift    115323 non-null  float64
     19  backsq      110517 non-null  float64
     20  pullups     50608 non-null   float64
     21  eat         93932 non-null   object 
     22  train       105831 non-null  object 
     23  background  98945 non-null   object 
     24  experience  104936 non-null  object 
     25  schedule    97875 non-null   object 
     26  howlong     109206 non-null  object 
    dtypes: float64(16), object(11)
    memory usage: 87.1+ MB
    


```python
df.head()
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
      <th>athlete_id</th>
      <th>name</th>
      <th>region</th>
      <th>team</th>
      <th>affiliate</th>
      <th>gender</th>
      <th>age</th>
      <th>height</th>
      <th>weight</th>
      <th>fran</th>
      <th>...</th>
      <th>snatch</th>
      <th>deadlift</th>
      <th>backsq</th>
      <th>pullups</th>
      <th>eat</th>
      <th>train</th>
      <th>background</th>
      <th>experience</th>
      <th>schedule</th>
      <th>howlong</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2554.0</td>
      <td>Pj Ablang</td>
      <td>South West</td>
      <td>Double Edge</td>
      <td>Double Edge CrossFit</td>
      <td>Male</td>
      <td>24.0</td>
      <td>70.0</td>
      <td>166.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>400.0</td>
      <td>305.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>I workout mostly at a CrossFit Affiliate|I hav...</td>
      <td>I played youth or high school level sports|I r...</td>
      <td>I began CrossFit with a coach (e.g. at an affi...</td>
      <td>I do multiple workouts in a day 2x a week|</td>
      <td>4+ years|</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3517.0</td>
      <td>Derek Abdella</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Male</td>
      <td>42.0</td>
      <td>70.0</td>
      <td>190.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>I have a coach who determines my programming|I...</td>
      <td>I played youth or high school level sports|</td>
      <td>I began CrossFit with a coach (e.g. at an affi...</td>
      <td>I do multiple workouts in a day 2x a week|</td>
      <td>4+ years|</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4691.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5164.0</td>
      <td>Abo Brandon</td>
      <td>Southern California</td>
      <td>LAX CrossFit</td>
      <td>LAX CrossFit</td>
      <td>Male</td>
      <td>40.0</td>
      <td>67.0</td>
      <td>NaN</td>
      <td>211.0</td>
      <td>...</td>
      <td>200.0</td>
      <td>375.0</td>
      <td>325.0</td>
      <td>25.0</td>
      <td>I eat 1-3 full cheat meals per week|</td>
      <td>I workout mostly at a CrossFit Affiliate|I hav...</td>
      <td>I played youth or high school level sports|</td>
      <td>I began CrossFit by trying it alone (without a...</td>
      <td>I usually only do 1 workout a day|</td>
      <td>4+ years|</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5286.0</td>
      <td>Bryce Abbey</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Male</td>
      <td>32.0</td>
      <td>65.0</td>
      <td>149.0</td>
      <td>206.0</td>
      <td>...</td>
      <td>150.0</td>
      <td>NaN</td>
      <td>325.0</td>
      <td>50.0</td>
      <td>I eat quality foods but don't measure the amount|</td>
      <td>I workout mostly at a CrossFit Affiliate|I inc...</td>
      <td>I played college sports|</td>
      <td>I began CrossFit by trying it alone (without a...</td>
      <td>I usually only do 1 workout a day|I strictly s...</td>
      <td>1-2 years|</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
msno.matrix(df, figsize=(14,6),fontsize=11);
```


    
![png](output_7_0.png)
    


Data that is not relevant to our analysis was removed from consideration along with the less frequently performed fitness events. It was necessary to remove these less common events to maintain a large dataset.


```python
df = df.dropna(subset=['region','age','weight','height','howlong','gender','eat','train','background','experience','schedule','howlong','deadlift','candj','snatch','backsq','experience','background','schedule','howlong']) #removing NaNs from parameters of interest 
df = df.drop(columns=['affiliate','team','name','athlete_id','fran','helen','grace','filthy50','fgonebad','run400','run5k','pullups','train']) #removing paramters not of interest + less popular events
```


```python
msno.matrix(df, figsize=(14,6),fontsize=11);
```


    
![png](output_10_0.png)
    


### Outlier Removal


```python
#removing problematic entries 
df = df[df['weight'] < 1500] #removes two anomolous weight entries of 1,750 and 2,113
df = df[df['gender']!='--'] #removes 9 non-male/female gender entries due to small sample size
df = df[df['age']>=18] #only considering adults 
df = df[(df['height']<96)&(df['height']>48)]#selects people between 4 and 8 feet

#no lifts above world recording holding lifts were included
df = df[(df['deadlift']>0)&(df['deadlift']<=1105)|((df['gender']=='Female')&(df['deadlift']<=636))] #removes negative deadlift weights and deadlifts above the current world record
df = df[(df['candj']>0)&(df['candj']<=395)]#|((df['gender']=='Female')&(df['candj']<=265))] #removes negative clean and jerk value and reported weights above the current world record
df = df[(df['snatch']>0)&(df['snatch']<=496)]#|((df['gender']=='Female')&(df['snatch']<=341))] #removes weights above the current world record 
df = df[(df['backsq']>0)&(df['backsq']<=1069)]#|((df['gender']=='Female')&(df['backsq']<=615))] #removes weights over current world record
```

### Cleaning survey data

Survey responses were cleaned and encoded. Problematically, multiple survey questions were collated into single responses on eating, background, training, and CrossFit experience. This means, for example, that if someone had worked as a CrossFit trainer, completed a CrossFit course, started doing CrossFit alone and declined to answer if they found CrossFit life changing, that all responses would be combined into one answer on experience. Similarly, it is possible for a single response to contain contradictory information. For instance, a respondent could indicate that they have no previous athletic background and were also a college athlete.

Thus, cleaning this data set, separating answers into individual question responses and removing non-sensical responses is non-trivial. Non-sensical survey responses were removed from consideration, and responses were organized to the best extent possible. 


```python
#get rid of declines to answer as only response 
decline_dict = {'Decline to answer|':np.nan}
df = df.replace(decline_dict)
df = df.dropna(subset=['background','experience','schedule','howlong','eat'])
```


```python
#encoding background data 

#encoding background questions 
df['rec'] = np.where(df['background'].str.contains('I regularly play recreational sports'), 1, 0)
df['high_school'] = np.where(df['background'].str.contains('I played youth or high school level sports'), 1, 0)
df['college'] = np.where(df['background'].str.contains('I played college sports'), 1, 0)
df['pro'] = np.where(df['background'].str.contains('I played professional sports'), 1, 0)
df['no_background'] = np.where(df['background'].str.contains('I have no athletic background besides CrossFit'), 1, 0)

#delete nonsense answers
df = df[~(((df['high_school']==1)|(df['college']==1)|(df['pro']==1)|(df['rec']==1))&(df['no_background']==1))] #you can't have no background and also a background 
```


```python
#encoding experience questions

#create encoded columns for experience reponse
df['exp_coach'] = np.where(df['experience'].str.contains('I began CrossFit with a coach'),1,0)
df['exp_alone'] = np.where(df['experience'].str.contains('I began CrossFit by trying it alone'),1,0)
df['exp_courses'] = np.where(df['experience'].str.contains('I have attended one or more specialty courses'),1,0)
df['life_changing'] = np.where(df['experience'].str.contains('I have had a life changing experience due to CrossFit'),1,0)
df['exp_trainer'] = np.where(df['experience'].str.contains('I train other people'),1,0)
df['exp_level1'] = np.where(df['experience'].str.contains('I have completed the CrossFit Level 1 certificate course'),1,0)

#delete nonsense answers
df = df[~((df['exp_coach']==1)&(df['exp_alone']==1))] #you can't start alone and with a coach

#creating no response option for coaching start
df['exp_start_nr'] = np.where(((df['exp_coach']==0)&(df['exp_alone']==0)),1,0)

#other options are assumed to be 0 if not explicitly selected
```


```python
#creating encoded columns with schedule data
df['rest_plus'] = np.where(df['schedule'].str.contains('I typically rest 4 or more days per month'),1,0)
df['rest_minus'] = np.where(df['schedule'].str.contains('I typically rest fewer than 4 days per month'),1,0)
df['rest_sched'] = np.where(df['schedule'].str.contains('I strictly schedule my rest days'),1,0)

df['sched_0extra'] = np.where(df['schedule'].str.contains('I usually only do 1 workout a day'),1,0)
df['sched_1extra'] = np.where(df['schedule'].str.contains('I do multiple workouts in a day 1x a week'),1,0)
df['sched_2extra'] = np.where(df['schedule'].str.contains('I do multiple workouts in a day 2x a week'),1,0)
df['sched_3extra'] = np.where(df['schedule'].str.contains('I do multiple workouts in a day 3\+ times a week'),1,0)

#removing/correcting problematic responses 
df = df[~((df['rest_plus']==1)&(df['rest_minus']==1))] #you can't have both more than and less than 4 rest days/month 

#points are only assigned for the highest extra workout value (3x only vs. 3x and 2x and 1x if multi selected)
df['sched_0extra'] = np.where((df['sched_3extra']==1),0,df['sched_0extra'])
df['sched_1extra'] = np.where((df['sched_3extra']==1),0,df['sched_1extra'])
df['sched_2extra'] = np.where((df['sched_3extra']==1),0,df['sched_2extra'])
df['sched_0extra'] = np.where((df['sched_2extra']==1),0,df['sched_0extra'])
df['sched_1extra'] = np.where((df['sched_2extra']==1),0,df['sched_1extra'])
df['sched_0extra'] = np.where((df['sched_1extra']==1),0,df['sched_0extra'])

#adding no response columns
df['sched_nr'] = np.where(((df['sched_0extra']==0)&(df['sched_1extra']==0)&(df['sched_2extra']==0)&(df['sched_3extra']==0)),1,0)
df['rest_nr'] = np.where(((df['rest_plus']==0)&(df['rest_minus']==0)),1,0)
#schedling rest days is assumed to be 0 if not explicitly selected
```


```python
# encoding howlong (crossfit lifetime)
df['exp_1to2yrs'] = np.where((df['howlong'].str.contains('1-2 years')),1,0)
df['exp_2to4yrs'] = np.where((df['howlong'].str.contains('2-4 years')),1,0)
df['exp_4plus'] = np.where((df['howlong'].str.contains('4\+ years')),1,0)
df['exp_6to12mo'] = np.where((df['howlong'].str.contains('6-12 months')),1,0)
df['exp_lt6mo'] = np.where((df['howlong'].str.contains('Less than 6 months')),1,0)

#keeping only higest repsonse 
df['exp_lt6mo'] = np.where((df['exp_4plus']==1),0,df['exp_lt6mo'])
df['exp_6to12mo'] = np.where((df['exp_4plus']==1),0,df['exp_6to12mo'])
df['exp_1to2yrs'] = np.where((df['exp_4plus']==1),0,df['exp_1to2yrs'])
df['exp_2to4yrs'] = np.where((df['exp_4plus']==1),0,df['exp_2to4yrs'])
df['exp_lt6mo'] = np.where((df['exp_2to4yrs']==1),0,df['exp_lt6mo'])
df['exp_6to12mo'] = np.where((df['exp_2to4yrs']==1),0,df['exp_6to12mo'])
df['exp_1to2yrs'] = np.where((df['exp_2to4yrs']==1),0,df['exp_1to2yrs'])
df['exp_lt6mo'] = np.where((df['exp_1to2yrs']==1),0,df['exp_lt6mo'])
df['exp_6to12mo'] = np.where((df['exp_1to2yrs']==1),0,df['exp_6to12mo'])
df['exp_lt6mo'] = np.where((df['exp_6to12mo']==1),0,df['exp_lt6mo'])
```


```python
#encoding dietary preferences 
df['eat_conv'] = np.where((df['eat'].str.contains('I eat whatever is convenient')),1,0)
df['eat_cheat']= np.where((df['eat'].str.contains('I eat 1-3 full cheat meals per week')),1,0)
df['eat_quality']= np.where((df['eat'].str.contains('I eat quality foods but don\'t measure the amount')),1,0)
df['eat_paleo']= np.where((df['eat'].str.contains('I eat strict Paleo')),1,0)
df['eat_cheat']= np.where((df['eat'].str.contains('I eat 1-3 full cheat meals per week')),1,0)
df['eat_weigh'] = np.where((df['eat'].str.contains('I weigh and measure my food')),1,0)
```


```python
#encoding location as US vs non-US
US_regions = ['Southern California', 'North East', 'North Central','South East', 'South Central', 'South West', 'Mid Atlantic','Northern California','Central East', 'North West']
df['US'] = np.where((df['region'].isin(US_regions)),1,0)
```


```python
#encoding gender
df['gender_'] = np.where(df['gender']=='Male',1,0)
```

## Feature Engineering

### Normalizing lifts

It is customary in powerlifting to normalize the weight lifted to the bodyweight of the athlete. This corrects for the physical advantage of leverage and isolates the athletic contribution to a lift.


```python
df['norm_dl'] = df['deadlift']/df['weight']
df['norm_j'] = df['candj']/df['weight']
df['norm_s'] = df['snatch']/df['weight']
df['norm_bs'] = df['backsq']/df['weight']

df['total_lift'] = df['norm_dl']+df['norm_j']+df['norm_s']+df['norm_bs']
```

### Calculating body mass index (BMI)
BMI is a measure of body composition based on weight and height. It categorizes individuals as underweight, normal weight, overweight, or obese, and can be used as a starting point to set fitness goals. However, BMI does not distinguish between muscle and fat and can mistake highly muscular individuals as overweight.


```python
df['BMI'] = df['weight']*0.453592/np.square(df['height']*0.0254)

df = df[(df['BMI']>=17)&(df['BMI']<=50)]#considers only underweight - morbidly obese competitors
```

# EDA

### Lifts


```python
plt.style.use('ggplot')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10,6
```


```python
xlabel = 'Lift in body weight'
plt.subplot(221)
sns.histplot(data = df, x = 'norm_dl', hue='gender',kde = True);
plt.title('Deadlift')
plt.xlabel(xlabel)
plt.subplot(222)
sns.histplot(data = df, x = 'norm_j', hue='gender',kde = True);
plt.title('Clean and jerk')
plt.xlabel(xlabel)
plt.tight_layout()
plt.subplot(223)
sns.histplot(data = df, x = 'norm_s', hue='gender',kde = True);
plt.title('Snatch')
plt.xlabel(xlabel)
plt.subplot(224)
sns.histplot(data = df, x = 'norm_bs', hue='gender',kde = True);
plt.title('Back squat')
plt.xlabel(xlabel)
plt.tight_layout()
```


    
![png](output_28_0.png)
    



```python
#total lift 
sns.histplot(data = df, x = 'total_lift', hue='gender',kde = True);
plt.title('Normalized total lift')
plt.xlabel(xlabel);
```


    
![png](output_29_0.png)
    


### Region


```python
plt.subplot(1,2,1)
plt.title('Region')
sns.histplot(data = df, x = 'region')
plt.xlabel('')
plt.xticks(rotation = 90)

plt.subplot(1,2,2)
plt.title('Country')
plt.bar('US',df['US'].sum())
plt.bar('non-US',(df.shape[0]-df['US'].sum()))

plt.tight_layout()
plt.show()
```


    
![png](output_31_0.png)
    



```python
plt.subplot(1,2,1)
sns.boxplot(x=df.region,y=df.total_lift)
plt.xticks(rotation = 90)
plt.xlabel('')
plt.ylabel('Total lift (bodyweight)')
ax = plt.subplot(1,2,2)
sns.boxplot(x=df.US, y=df.total_lift, hue = df.gender)
plt.tight_layout()
plt.ylabel('Total lift (bodyweight)')
ax.set_xticklabels(('Outside US','US'))
plt.xlabel('')
plt.show()
```


    
![png](output_32_0.png)
    


### BMI


```python
#Mean BMI
def standard_error(x):
    stdev = x.std()
    counts = x.count()
    return stdev/np.sqrt(counts)

df['bmi_rounded'] =  df['BMI'].round()
df_bmi = df[['bmi_rounded','total_lift','gender']].groupby(['bmi_rounded','gender']).agg(['mean',standard_error,'size']).sort_values('bmi_rounded').reset_index()
df_bmi_f = df_bmi[(df_bmi['gender']=='Female')&(df_bmi['total_lift']['size']>5)]
df_bmi_m = df_bmi[(df_bmi['gender']=='Male')&(df_bmi['total_lift']['size']>5)]
```


```python
plt.subplot(1,2,1)
sns.histplot(data=df, x='height', hue='gender', binwidth=1,kde=True)
plt.subplot(1,2,2)
sns.histplot(data=df, x='weight', hue='gender', binwidth=5, kde=True)
plt.show()
```


    
![png](output_35_0.png)
    



```python
sns.histplot(data = df, x='BMI',hue='gender',binwidth=1,kde=True)
plt.show()
```


    
![png](output_36_0.png)
    



```python
plt.subplot(1,2,1)
plt.scatter(df.BMI[df.gender=='Male'], df.total_lift[df.gender=='Male'],alpha=0.01,color='crimson',label='Male')
plt.scatter(df.BMI[df.gender=='Female'], df.total_lift[df.gender=='Female'],alpha=0.01,color='teal',label='Female')
plt.xlabel('BMI (kg/m2)')
plt.ylabel('Total lift \n(body weight)')
leg = plt.legend()
for lh in leg.legendHandles: 
    lh.set_alpha(1)


plt.subplot(1,2,2)
plt.plot(df_bmi_m['bmi_rounded'],df_bmi_m['total_lift']['mean'],'.-',color='crimson',label='Male')
plt.errorbar(df_bmi_m['bmi_rounded'],df_bmi_m['total_lift']['mean'],yerr=df_bmi_m['total_lift']['standard_error'],color='crimson',label='_male_')
plt.plot(df_bmi_f['bmi_rounded'],df_bmi_f['total_lift']['mean'],'.-',color='teal',label='Female')
plt.errorbar(df_bmi_f['bmi_rounded'],df_bmi_f['total_lift']['mean'],yerr=df_bmi_f['total_lift']['standard_error'],color='teal',label='_female_')
plt.xlabel('BMI (kg/m2)')
plt.ylabel('Mean total lift \n(bodyweight)')
plt.legend()

plt.tight_layout()
```


    
![png](output_37_0.png)
    


The normalized total weight lifted for each competitor is shown in the figure on the left against BMI. Male athletes tend towards higher BMIs and lift more weight. Averaged data is shown in the right plot with standard error and indicates performance peaks at BMIs of 23 kg/m2 and 27 kg/m2 for female and male athletes respectively. Likely, this represents athletes gaining more muscle and eventually more fat, as more muscle would help performance and more fat would adversely affect performance. Thus, this analysis suggests that male and female athletes should adjust their training and diet to reach BMIs of 27 and 23 respectively with lean body compositions. 

### Age


```python
sns.histplot(data = df, x = 'age', hue='gender',binwidth=1,kde = True)
plt.show()
```


    
![png](output_40_0.png)
    



```python
df_age = df[['age','gender','total_lift']].groupby(['age','gender']).agg(['mean',standard_error,'size']).sort_values('age').reset_index()
df_age_f = df_age[(df_age['gender']=='Female')&(df_age['total_lift']['size']>5)]
df_age_m = df_age[(df_age['gender']=='Male')&(df_age['total_lift']['size']>5)]

plt.subplot(1,2,1)
plt.scatter(df.age[df.gender=='Male'], df.total_lift[df.gender=='Male'],alpha=0.01,color='crimson',label='Male')
plt.scatter(df.age[df.gender=='Female'], df.total_lift[df.gender=='Female'],alpha=0.01,color='teal',label='Female')
plt.xlabel('Years')
plt.ylabel('Total lift \n(bodyweight)')
leg = plt.legend()
for lh in leg.legendHandles: 
    lh.set_alpha(1)

plt.subplot(1,2,2)
sns.lineplot(x=df_age_m['age'],y=df_age_m['total_lift']['mean'],color='crimson',label='Male')
plt.errorbar(df_age_m.age,df_age_m['total_lift']['mean'],yerr=df_age_m['total_lift']['standard_error'],color='crimson',label='_Male_')
sns.lineplot(x=df_age_f['age'],y=df_age_f['total_lift']['mean'],color='teal', label='Female')
plt.errorbar(df_age_f.age,df_age_f['total_lift']['mean'],yerr=df_age_f['total_lift']['standard_error'],color='teal',label='_Female_')
plt.legend()
plt.xlabel('Years')
plt.ylabel('Mean total lift \n(bodyweight)')

plt.tight_layout()
```


    
![png](output_41_0.png)
    


The normalized total lift for all athletes is shown against age in the figure on the left. Male competitors lift slightly more, and a general decrease in performance past 25 is apparent. This data is averaged and shown with standard error on the plot on the right. Male performance shows a clear peak around 20 years of age, and female performance maintains a plateau until around 30 years of age. Both populations show distinct decreases in performance with increased age.

Interestingly, male performance appears to be more sensitive to age than female performance. Testosterone is extremely important to building and maintaining muscle and strength, and it is possible that declines in testosterone characteristic to male aging underly this observation.

### Athlete lifestyle 


```python
plt.figure(figsize=(14,9))
plt.title('Lifestyle Survey Responses')
plt.subplot(2,3,1)
plt.bar('<0.5',df['exp_lt6mo'].sum())
plt.bar('0.5-1',df['exp_6to12mo'].sum())
plt.bar('1-2',df['exp_1to2yrs'].sum())
plt.bar('2-4',df['exp_2to4yrs'].sum())
plt.bar('4+',df['exp_4plus'].sum())
plt.title('Crossfit age');
plt.ylabel('Count')
plt.xlabel('Years')

plt.subplot(2,3,2)
plt.bar('0', df['sched_0extra'].sum())
plt.bar('1', df['sched_1extra'].sum())
plt.bar('2', df['sched_2extra'].sum())
plt.bar('3+', df['sched_3extra'].sum())
plt.title('Two-a-day frequency')
plt.ylabel('Count')
plt.xlabel('Frequency per week')
plt.xticks(rotation = 90);

plt.subplot(2,3,3)
#plt.bar('Scheduled', df['rest_sched'].sum())
plt.bar('4+', df['rest_plus'].sum())
plt.bar('<4 ', df['rest_minus'].sum())
plt.title('Rest days')
plt.xlabel('Days per month')
plt.ylabel('Count')

plt.subplot(2,3,4)
plt.title('Started with coach')
plt.bar('No', df['exp_alone'].sum()-df['exp_start_nr'].sum())
plt.bar('Yes', df['exp_coach'].sum())
plt.ylabel('Count')
#plt.bar('Crossfit trainer', df['trainer'].sum())
#plt.bar('Completed specialty course', df['specialty_course'].sum())
#plt.bar('Completed level 1', df['level_one'].sum())
#plt.xticks(rotation=90);

plt.subplot(2,3,5)
plt.bar('High \nschool', df['high_school'].sum())
plt.bar('College', df['college'].sum())
plt.bar('Pro', df['pro'].sum())
plt.bar('None', df['no_background'].sum())
plt.bar('Rec. \nsports', df['rec'].sum())
plt.ylabel('Count')
#plt.xlabel('Background')
plt.title('Athletic history')
#plt.xticks(rotation=90)

plt.subplot(2,3,6)
plt.bar('Eats \nquality', df['eat_quality'].sum())
plt.bar('Cheat \nmeals', df['eat_cheat'].sum())
plt.bar('Measures \nfood', df['eat_weigh'].sum())
plt.bar('Paleo', df['eat_paleo'].sum())
plt.bar('Eats \nconvenience', df['eat_conv'].sum())
plt.ylabel('Count')
plt.title('Eating habits')
#plt.xticks(rotation=90)

plt.tight_layout()
```


    
![png](output_44_0.png)
    



```python
#melting data for boxplots
df_exp =  pd.melt(df, id_vars=['total_lift'], var_name='experience', value_vars=['exp_lt6mo','exp_6to12mo','exp_1to2yrs','exp_2to4yrs','exp_4plus','total_lift'])
df_exp = df_exp[df_exp!=0].dropna()
df_freq = pd.melt(df, id_vars=['total_lift'], var_name='freq', value_vars=['sched_0extra','sched_1extra','sched_2extra','sched_3extra'])
df_freq = df_freq[df_freq!=0].dropna()
df_rest = pd.melt(df, id_vars=['total_lift'], var_name='rest', value_vars=['rest_plus', 'rest_minus'])
df_rest = df_rest[df_rest!=0].dropna()
df_hist = pd.melt(df, id_vars=['total_lift'], var_name='hist', value_vars=['high_school','college','pro','no_background','rec'])
df_hist = df_hist[df_hist!=0].dropna()
df_start = pd.melt(df, id_vars=['total_lift'], var_name='start', value_vars=['exp_alone','exp_coach'])
df_start = df_start[df_start!=0].dropna()
df_eat = pd.melt(df, id_vars=['total_lift'], var_name='eat', value_vars=['eat_quality','eat_cheat','eat_weigh','eat_paleo','eat_conv'])
df_eat = df_eat[df_eat!=0].dropna()

y_label = 'Body weights lifted (total)'

#box plots of above figures 
fig, axs = plt.subplots(nrows=2, ncols=3, sharey=True, figsize=(14,9))

for ax in axs.flatten()[1:]:
    ax.yaxis.label.set_visible(False)
    #ax.yaxis.set_ticks([])
    
ax1 = plt.subplot(2,3,1)
sns.boxplot(data = df_exp, x='experience',y='total_lift')
ax1.set_xticklabels(('<0.5','0.5-1','1-2','2-4','4+'))
ax1.set_xlabel('Years')
ax1.set_ylabel(y_label)
ax1.set_title('Crossfit age')
ax2 = plt.subplot(2,3,2)    
sns.boxplot(data=df_freq, x='freq',y='total_lift')
ax2.set_xticklabels(('0','1','2','3+'))
ax2.set_xlabel('Frequency per week')
ax2.set_title('Two-a-day frequency')
ax3 = plt.subplot(2,3,3)    
sns.boxplot(data=df_rest, x='rest',y='total_lift')
ax3.set_xticklabels(('<4','4+'))
ax3.set_title('Rest days')
ax3.set_xlabel('Days per month')
ax4 = plt.subplot(2,3,4) 
sns.boxplot(data=df_start,x='start',y='total_lift')
ax4.set_xticklabels(('No','Yes'))
ax4.set_xlabel('')
ax4.set_title('Started with coach')
ax4.yaxis.label.set_visible(True)
ax4.set_ylabel(y_label)
ax5 = plt.subplot(2,3,5)    
sns.boxplot(data=df_hist, x='hist',y='total_lift')
ax5.set_xticklabels(('High \nschool','College','Pro','None','Rec. \nsports'))
ax5.set_xlabel('')
ax5.set_title('Athletic history')
ax6 = plt.subplot(2,3,6)    
sns.boxplot(data=df_eat, x='eat',y='total_lift')
ax6.set_xticklabels(('Eats \nquality','Cheat \nmeals','Measures \nfood','Paleo','Eats \nconvenience'))
ax6.set_xlabel('')
ax6.set_title('Eating habits')
plt.tight_layout()
```


    
![png](output_45_0.png)
    


CrossFit age (years doing CrossFit), frequency of two-a-day session, taking sufficient rest days, and athletic background all correlate with improved athletic performance as would be expected from conventional knowledge. Interestingly, starting CrossFit alone rather than at a facility with a coach appears to also correlate with improved performance. Reported eating habits, however, do not appear strongly correlated with performance.

### Has CrossFit been life changing?


```python
changed = df['life_changing'].sum()
not_changed = df.shape[0]-changed #non-answers are considered non-life changing responses

plt.subplot(1,2,1)
plt.pie([not_changed, changed],labels=['Not life changing','Life changing'])

plt.subplot(1,2,2)
sns.boxplot(x = df.life_changing, y=df.total_lift)
plt.ylabel('Total Lift (bodyweight)')
plt.xlabel('Is crossfit lifechanging?')
plt.tight_layout()
```


    
![png](output_48_0.png)
    


Seeing crossfit as a life changing event, likely represents strong athelte enthusiasim. However, this metric does not apperar strongly correlated to athlete performance. It seems that enjoying the ritual and community of crossfit may not be linked to athletic performance.

### Correlation matrix


```python
#Heat map 

#optimizng range for color scale
vmin = df.corr().min().min()
vmax = df.corr()[df.corr()!=1].max().max()

#thresholding selected correlations
df_corr  = df.drop(columns=['backsq','deadlift','candj','snatch','exp_start_nr','sched_nr','rest_nr']).corr()[np.absolute(df.corr())>0.2]

#Mask for selecting only bottom triangle
mask = np.triu(df_corr)

plt.figure(figsize=(15,10))
#plt.style.context('default')
sns.heatmap(df_corr,vmin=vmin, vmax=vmax,mask=mask,cmap='seismic')
plt.grid()
#print(mask)
```


    
![png](output_51_0.png)
    


The Pearson correlation coefficient is a statistical measure that quantifies the strength and direction of the linear relationship between two continuous variables. It ranges from -1 to 1, where a value of 1 indicates a perfect positive correlation, -1 indicates a perfect negative correlation, and 0 indicates no linear correlation. Frequency of two-a-day training sessions, completing a level 1 certification, starting CrossFit alone, and working as a CrossFit trainer are all correlated to improved performance, while having no athletic background prior to CrossFit is correlated to worse performance. 

The strong non-linear correlations between performance and BMI and age are not captured in the above matrix. It is important to note that the Pearson correlation coefficient only measures linear relationships and may not capture non-linear associations or other types of relationships between variables.

# Predicting normalized total lift


```python
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
```


```python
df_select = df.drop(columns=['region','height','weight','candj','snatch','deadlift','norm_bs', 'norm_dl', 'norm_j', 'norm_s','bmi_rounded','backsq','eat','background','experience','schedule','howlong','gender'])
print('Selected features:\n',df_select.columns.values)
```

    Selected features:
     ['age' 'rec' 'high_school' 'college' 'pro' 'no_background' 'exp_coach'
     'exp_alone' 'exp_courses' 'life_changing' 'exp_trainer' 'exp_level1'
     'exp_start_nr' 'rest_plus' 'rest_minus' 'rest_sched' 'sched_0extra'
     'sched_1extra' 'sched_2extra' 'sched_3extra' 'sched_nr' 'rest_nr'
     'exp_1to2yrs' 'exp_2to4yrs' 'exp_4plus' 'exp_6to12mo' 'exp_lt6mo'
     'eat_conv' 'eat_cheat' 'eat_quality' 'eat_paleo' 'eat_weigh' 'US'
     'gender_' 'total_lift' 'BMI']
    

Features are selected for use in predictive models. Features that are redundant, have been encoded, or have been engineered into new features are not included in further analyses. Individual event performances are no longer considered, and the target is the total weight lifted normalized by athlete bodyweight.

### Random forest regression model

A random forest regression model is optimized to predict athlete performance. The random forest regressor randomly selects subsets of the original dataset with replacement, creating multiple training sets known as bootstrap samples. For each bootstrap sample, a decision tree is constructed using a subset of features. The decision tree is built by recursively splitting the data based on the selected features and their optimal thresholds. The final prediction is obtained by aggregating or "bagging" the individual predictions made by each decision tree. Random forest regression models work well with tabular data, large datasets, high-dimensional data, and non-linear data. 

In our modeling we use the root mean squared error is used as a cost function: 

<img src="https://miro.medium.com/max/327/1*9hQVcasuwx5ddq_s3MFCyw.gif" />



```python
#Assigning test and train sets
train_set, test_set = train_test_split(df_select, test_size=0.2, random_state=10)

X_train, y_train = train_set.drop(columns=['total_lift']), train_set['total_lift']
X_test, y_test = test_set.drop(columns=['total_lift']), test_set['total_lift']
```


```python
#random forest baseline
rnd_clf = RandomForestRegressor(n_estimators=100, max_depth = 12,oob_score=True, random_state=10).fit(X_train,y_train)
rnd_pred = rnd_clf.predict(X_test)
rnd_score = mean_squared_error(y_test, rnd_pred,squared=False)
print(f'Random forest regression model baseline RMSE: \n{rnd_score.round(3)}')
```

    Random forest regression model baseline RMSE: 
    0.863
    

### XGBoost

XGBoost stands for eXtreme gradient boosting and is a gradient boosted trees algorithm. In boosting, trees are sequentially built such that each subsequent tree aims to reduce the errors of the previous tree. Custom XGBoost callbacks were created during model optimizaiton to control learning rate decay, implement early stopping, and plot training results. A schematic of this iterative process is shown in the figure below.
<br>
<img src="https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/xgboost_illustration.png" />
</br>
<br>
Image from [AWS, How XGBoost Works](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-HowItWorks.html)
</br>

In addition, several features of XGBoost often lead to superior performance with tabular data. For example, XGBoost grows trees up to a specifiable maximum depth and prunes backwards to improve model fit. This is unlike other algorithms that build trees from the top down and stop once a negative loss is encountered on a single splitting step using a "greedy" algorithm. Additionally, there is built in regularization to avoid overfitting and capacity for parallelization.



```python
#defining custom learning rate decay
def learning_rate_decay(boosting_round): #, num_boost_round):
    learning_rate_start = 0.4
    learning_rate_min = 0.05
    lr_decay = 0.7
    lr = learning_rate_start * np.power(lr_decay, boosting_round)
    return max(learning_rate_min, lr)

lr_callback = xgb.callback.LearningRateScheduler(learning_rate_decay)
```


```python
#defining callback for plotting XGBoost fit progress, adapting from XGBoost documentation
class Plotting(xgb.callback.TrainingCallback):
    def __init__(self, rounds):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.rounds = rounds
        self.lines = {}
        self.fig.show()
        self.x = np.linspace(0, self.rounds, self.rounds)
        plt.ion()

    def _get_key(self, data, metric):
        return f'{data}-{metric}'

    def after_iteration(self, model, epoch, evals_log):
        if not self.lines:
            for data, metric in evals_log.items():
                for metric_name, log in metric.items():
                    key = self._get_key(data, metric_name)
                    expanded = log + [0] * (self.rounds - len(log))
                    self.lines[key],  = self.ax.plot(self.x, expanded, label=key)
                    self.ax.legend()
        else:
            for data, metric in evals_log.items():
                for metric_name, log in metric.items():
                    key = self._get_key(data, metric_name)
                    expanded = log + [0] * (self.rounds - len(log))
                    self.lines[key].set_ydata(expanded)
            self.fig.canvas.draw()
        return False
```


```python
#defining early stopping callback
es = xgb.callback.EarlyStopping(rounds=100,save_best=True)
```


```python
#XGBoost 
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
param = {'max_depth': 4,'subsample':.6,'reg_alpha':0}
evallist = [(dtrain, 'train'), (dtest, 'eval')]
num_round = 1500
print('Training and evluation RMSE scores:')
#bst = xgb.train(param, dtrain, num_round, evallist, verbose_eval=100,early_stopping_rounds=100,callbacks=[lr_callback, Plotting(num_round)])
bst= xgb.train(param, dtrain, num_round, evals=evallist, verbose_eval=100,callbacks=[lr_callback, Plotting(num_round),es])
plt.xlabel('Training round')
plt.ylabel('RMSE');
plt.title('Training XGBoost')
plt.xlim((0,bst.best_iteration));
```

    Training and evluation RMSE scores:
    [0]	train-rmse:3.80923	eval-rmse:3.81266
    [100]	train-rmse:0.82707	eval-rmse:0.85081
    [200]	train-rmse:0.81175	eval-rmse:0.84634
    [300]	train-rmse:0.80052	eval-rmse:0.84445
    [400]	train-rmse:0.79081	eval-rmse:0.84269
    [500]	train-rmse:0.78215	eval-rmse:0.84173
    [600]	train-rmse:0.77440	eval-rmse:0.84027
    [700]	train-rmse:0.76668	eval-rmse:0.83956
    [800]	train-rmse:0.75950	eval-rmse:0.83924
    [900]	train-rmse:0.75302	eval-rmse:0.83926
    [1000]	train-rmse:0.74687	eval-rmse:0.83898
    [1100]	train-rmse:0.74078	eval-rmse:0.83858
    [1123]	train-rmse:0.73955	eval-rmse:0.83868
    


    
![png](output_64_1.png)
    



```python
bst_pred = bst.predict(dtest)
bst_rmse = mean_squared_error(y_test, bst_pred,squared=False)

print('Best iteration and score:')
bst.best_iteration, round(bst_rmse,3)
```

    Best iteration and score:
    




    (1024, 0.839)




```python
print('Improvement over random forest regression model:')
improvement = (rnd_score - bst_rmse)/rnd_score *100
print(str(round(improvement,1))+'%')
```

    Improvement over random forest regression model:
    2.8%
    


```python
xgb.plot_tree(bst, num_trees=0);
```


    
![png](output_67_0.png)
    


### Feature Importance
One benefit of tree-based models is the ability to extract the predictive importance of input features. For random forest models, the built-in feature importance metric measures the average improvement to a prediction when each feature is used as a splitting node. For XGBoost models, weight, gain or cover metrics are built-in feature importance metrics. Gain is most like the random forest feature importance metric, while weight represents the relative number of times a feature is used as a splitting node and cover represents the number of observations split by a feature. While potentially insightful, these metrics tend to inflate the importance of continuous or high-cardinality categorical variables. 

Permutation importance algorithms evaluate the importance of a feature by comparing the performance of the baseline model to the performance with each feature permuted. This strategy does not require sequentially retraining the model necessarily but instead re-runs a trained model with each input feature permutated. The built-in Scikit-learn permutation importance function cannot be used with XGBoost. Thus, a custom permutation algorithm is developed. 


```python
def feature_importance(bst, X_test, y_test):
    baseline_d = xgb.DMatrix(X_test, label=y_test)
    y_baseline = bst.predict(baseline_d)
    baseline = mean_squared_error(y_test, y_baseline,squared=False)
    
    f_imp = []
    
    for col in X_test.columns:
        save = X_test[col].copy()
        X_test[col] = np.random.permutation(X_test[col])

        perm_d = xgb.DMatrix(X_test, label=y_test)
        y_perm = bst.predict(perm_d)
        perm_baseline = mean_squared_error(y_test, y_perm, squared=False)
        f_imp.append(perm_baseline-baseline)
        
        X_test[col] = save #resets values
       
    return pd.DataFrame(data = f_imp, index = X_test.columns.values).sort_values(0,ascending=False).plot(kind='bar',ylabel = 'Decrease in RMSE from baseline', title='Feature Importance by permutation',legend=False)
```


```python
feature_importance(bst, X_test, y_test);
```


    
![png](output_70_0.png)
    


The most significant decreases to the RMSE were observed when gender, age and BMI were permutated. This is in line with strong the correlations observed in our EDA with these features and performance. Secondary features of importance were CrossFit age, frequency of extra training sessions, athletic background, and region. Interestingly, taking rest days and all dietary habits were not of consequence to the model. It is possible that these behaviors don't significantly affect performance.

# Dense neural network

A dense neural network was created to predict CrossFit athlete performance.  Neural networks are useful in situations where identifying complex patterns aids in predictive performance. Neural networks have been outperformed by XGBoost when working with smaller amounts of data and tabular data. 


```python
import tensorflow as tf
from keras.layers import Dropout, BatchNormalization
from functools import partial 
```


```python
#splitting test data to make a validation dataset
valid_set_nn, test_set_nn = train_test_split(test_set, test_size=0.5, random_state=10)
X_valid_nn = valid_set_nn.drop(columns=['total_lift'])
y_valid_nn = valid_set_nn['total_lift'].values.reshape(-1,1)
X_test_nn = test_set_nn.drop(columns=['total_lift'])
y_test_nn = test_set_nn['total_lift']
```


```python
#building neural network
tf.random.set_seed(10)

layer_size = 300
RegularizedDense = partial(tf.keras.layers.Dense,
                           activation = 'relu',
                           kernel_initializer='he_normal')
                           #kernel_regularizer = tf.keras.regularizers.l2(0.001))

model = tf.keras.Sequential([
    tf.keras.Input(shape=[X_train.shape[1]]),
    BatchNormalization(name = 'Batch_Normalization'),
    RegularizedDense(layer_size, name = "Dense_1"),
    Dropout(0.3, name = 'Dropout_1'),
    RegularizedDense(layer_size, name = "Dense_2"),
    Dropout(0.3, name= 'Dropout_2'),
    tf.keras.layers.Dense(1, activation='relu', name = 'Dense_3')
])

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.005,
    decay_steps=30,
    decay_rate=0.99)
    
opt = tf.keras.optimizers.experimental.AdamW(learning_rate=lr_schedule)
                   
model.compile(loss='mse',
              optimizer = opt,
              metrics=['RootMeanSquaredError']
             )

model.training=True
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     Batch_Normalization (BatchN  (None, 35)               140       
     ormalization)                                                   
                                                                     
     Dense_1 (Dense)             (None, 300)               10800     
                                                                     
     Dropout_1 (Dropout)         (None, 300)               0         
                                                                     
     Dense_2 (Dense)             (None, 300)               90300     
                                                                     
     Dropout_2 (Dropout)         (None, 300)               0         
                                                                     
     Dense_3 (Dense)             (None, 1)                 301       
                                                                     
    =================================================================
    Total params: 101,541
    Trainable params: 101,471
    Non-trainable params: 70
    _________________________________________________________________
    


A shallow neural network with two hidden layers was created to predict the normalized total lift. The number of neurons in each hidden layer was adjusted upwards until the model overfit the training data, and then regularization techniques were added to prevent overfitting. Dropout was the most effective regularization method sampled, and was used over kernel regularization, gradient clipping, and other techniques. It is possible that the ensemble effect observed with dropout was particularly beneficial to this dataset. The final model had 300 neurons in each of two hidden layers and two dropout layers.   



```python
history = model.fit(X_train, y_train, epochs=30, verbose=False, validation_data=(X_valid_nn, y_valid_nn))
```


```python
pd.DataFrame(history.history).plot(xlabel='Epoch',ylim=[0,2],ylabel='Metric');
```


    
![png](output_78_0.png)
    



```python
model.training=False
y_pred_nn = model.predict(X_test_nn, verbose=False)
nn_rmse = mean_squared_error(y_test_nn, y_pred_nn,squared=False)
print('Neural network RMSE:')
round(nn_rmse,3)
```

    Neural network RMSE:
    




    0.863



XGBoost outperformed the shallow neural network on this dataset. The neural network model had a higher bias than the tree-based models. 

### Benchmarking model performance

To benchmark model performance, 10 random participants were selected. After careful study and review, I predicted each participant’s normalized total lift from the input features. The RMSE of my predictions is calculated for comparison to the other models. Optimally, the predictions of a team of CrossFit experts and coaches would be used, but lacking these connections and resources my judgement will be used to estimate human-level predictive power. 


```python
#selecting index 
benchmark = df_select.sample(n=10, random_state=10)
benchmark_X = benchmark.drop(columns=['total_lift'])
benchmark_y = benchmark['total_lift']
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 10)
benchmark_X
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
      <th>age</th>
      <th>rec</th>
      <th>high_school</th>
      <th>college</th>
      <th>pro</th>
      <th>no_background</th>
      <th>exp_coach</th>
      <th>exp_alone</th>
      <th>exp_courses</th>
      <th>life_changing</th>
      <th>exp_trainer</th>
      <th>exp_level1</th>
      <th>exp_start_nr</th>
      <th>rest_plus</th>
      <th>rest_minus</th>
      <th>rest_sched</th>
      <th>sched_0extra</th>
      <th>sched_1extra</th>
      <th>sched_2extra</th>
      <th>sched_3extra</th>
      <th>sched_nr</th>
      <th>rest_nr</th>
      <th>exp_1to2yrs</th>
      <th>exp_2to4yrs</th>
      <th>exp_4plus</th>
      <th>exp_6to12mo</th>
      <th>exp_lt6mo</th>
      <th>eat_conv</th>
      <th>eat_cheat</th>
      <th>eat_quality</th>
      <th>eat_paleo</th>
      <th>eat_weigh</th>
      <th>US</th>
      <th>gender_</th>
      <th>BMI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>196036</th>
      <td>27.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>20.254429</td>
    </tr>
    <tr>
      <th>158151</th>
      <td>18.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>26.608364</td>
    </tr>
    <tr>
      <th>81732</th>
      <td>23.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>25.632724</td>
    </tr>
    <tr>
      <th>13402</th>
      <td>31.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>28.889081</td>
    </tr>
    <tr>
      <th>61466</th>
      <td>36.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>43.941813</td>
    </tr>
    <tr>
      <th>4714</th>
      <td>33.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>28.696694</td>
    </tr>
    <tr>
      <th>259</th>
      <td>27.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>24.394286</td>
    </tr>
    <tr>
      <th>155078</th>
      <td>39.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>28.128842</td>
    </tr>
    <tr>
      <th>126860</th>
      <td>34.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>27.046190</td>
    </tr>
    <tr>
      <th>103760</th>
      <td>20.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>28.974775</td>
    </tr>
  </tbody>
</table>
</div>




```python
#my guesses
my_y = [5.5,7,6.5,6,4,6,8,4,5,6]
#my RMSE
my_rmse = mean_squared_error(benchmark_y, my_y,squared=False)
print('My RMSE:')
round(my_rmse, 3)
```

    My RMSE:
    




    1.782




```python
#Summary of model performance
p = pd.DataFrame([rnd_score, bst_rmse, nn_rmse, my_rmse],index=['Random Forest','XGBoost','Neural network','Me'],columns=['RMSE']).T.round(3)
p
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
      <th>Random Forest</th>
      <th>XGBoost</th>
      <th>Neural network</th>
      <th>Me</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>RMSE</th>
      <td>0.863</td>
      <td>0.839</td>
      <td>0.863</td>
      <td>1.782</td>
    </tr>
  </tbody>
</table>
</div>



All models outperformed my estimation of athlete normalized total lift. While the models were unable to perfectly predict performance, they are better tools than me and likely even other experts at predicting performance.

# Conclusions and Recommendations

### Advice for athletes and coaches
The results of this investigation indicate that age, BMI and gender are the key metrics for predicting CrossFit athlete performance. Unfortunately for athletes and coaches, age and gender are set and cannot be adjusted through training. However, body composition can be changed through careful training and diet. Our study indicates that male and female CrossFit athletes should target lean body compositions and BMIs of 23 kg/m2 and 27 mg/m2 respectively for optimal performance. 

Other features that positively impact performance include CrossFit age, frequency of two-a-day training sessions, and taking rest days. Athletes should be encouraged to be patient, as more experience with CrossFit is correlated with improved performance. Similarly, adding additional training sessions boosts performance up to the surveyed limit of three or more two-a-day sessions per week without any apparent ill effect from overtraining. Athletes should also be encouraged to take ~1 rest day per week, as this will also boost performance. 
Interestingly, eating habits as surveyed do not appear to impact athlete performance. Adhering to a paleo diet is strongly encouraged for CrossFit athletes. However, this does not appear to be correlated with improved lifts or help our predictive models. Thus, eating paleo may not be important for competitor success. 

### Survey suggestions 
To improve the usefulness and predictive power of future surveys, the following improvements are suggested:

-	Encode one survey question response per answer rather than collating multiple thematically aligned answers into one response to avoid ambiguity in response interpretation. 

-	Encourage athletes to complete more events, as gymnastic-style and cardio intensive events were not included due to the lack of data. Thus, our analysis was limited to strength. Adding cardio and gymnastics may change the training recommendations given. 


-	Add questions on protein consumption, specifically if protein consumption is tracked and average daily intake if tracked. Protein consumption is known to aid in muscle development and retention and may be more relevant to performance.  Current survey questions on dietary habits may not be useful and do not need to be included in future surveys. 

<br>
<br>