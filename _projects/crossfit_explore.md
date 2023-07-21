---
layout: page
title: A data-based approach to CrossFit training
description: CrossFit game data is explored and analyzed and used to improve athlete preparation.
img: assets/img/project_previews/crossfit_explore.png
importance: 2
category: Scikit-learn
---

# A data-based approach to CrossFit training 

<h2>Project Overview</h2>

CrossFit is a high-intensity fitness program that combines elements of weightlifting, cardio, and gymnastics. It aims to improve overall physical fitness by incorporating constantly varied functional movements performed at a high intensity. At the pinnacle of CrossFit is the CrossFit Games, an annual competition that showcases the world's fittest athletes. The CrossFit Games serve as a platform for elite athletes to test their skills and compete in a wide range of demanding workouts, challenging their strength, speed, power, and mental resilience. In this analysis, we will delve into the performance of CrossFit athletes, examining key factors that contribute to their success in this highly demanding and competitive sport.  

In this comprehensive review of survey data from the CrossFit games, we first clean highly disorganized survey data profiling athletes competing in the CrossFit games. This data encompasses age, height, weight, training habits (including rest days and sessions per day), geographic location, eating habits (such as measuring food, adhering to paleo or quality eating practices), and performance data. Features are engineered for a more insightful analysis of athlete performance, enabling us to uncover valuable trends and patterns. We then look at how these factors correlate with athlete performance to gain valuable insight for athletes, coaches and future survey construction. 

The dataset used in this project was provided by Ulrik Pedersen and can be found on [Kaggle](https://www.kaggle.com/datasets/ulrikthygepedersen/crossfit-athletes).

<h2><br></h2>
<h2>Project outline</h2>

 - Data overview 
 - Data cleaning 
    - Event selection
    - Outlier selection 
    - Cleaning survey Data
- Feature engineering 
    - Normalizing lifts
    - Calculating body mass index (BMI)
- Event performance
- Athlete demographics and performance 
    - Gender
    - BMI
    - Age 
    - Region 
- Lifestyle factors
- Has CrossFit been life changing?
- Feature correlation 
- Conclusions and recommendations  

<h2><br></h2>
<h2>Data overview</h2>


```python
import pandas as pd
import missingno as msno
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
```


```python
os.chdir('C:\\Users\\corne\\OneDrive\\Documents\\DS_Portfolio\\crossfit_project\\crossfit_project')
df = pd.read_csv('CrossFit_Games_data.csv')
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


    
![missing data](\assets\img\crossfit_explore\output_5_0.png)
    

<h2><br></h2>
<h2>Data cleaning</h2>

### Event selection
Data that is not relevant to our analysis was removed from consideration along with the less frequently performed fitness events. It was necessary to remove these less common events to maintain a large dataset.


```python
df = df.dropna(subset=['region','age','weight','height','howlong','gender','eat','train','background','experience','schedule','howlong','deadlift','candj','snatch','backsq','experience','background','schedule','howlong']) #removing NaNs from parameters of interest 
df = df.drop(columns=['affiliate','team','name','athlete_id','fran','helen','grace','filthy50','fgonebad','run400','run5k','pullups','train']) #removing paramters not of interest + less popular events
```


```python
msno.matrix(df, figsize=(14,6),fontsize=11);
```


    
![missing data](\assets\img\crossfit_explore\output_8_0.png)
    


### Outlier Removal

Only clear outliers were removed from the dataset. These outliers are suspected to be the result of data entry errors and were diagnosed from highly inprobable heights and weights and performances in events exceeding current world records. Only adults and male and female competitors were considered in this analysis.


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
<h2><br></h2>
<h2>Feature Engineering</h2>

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
<h2><br></h2>
<h2>Event performance</h2>

Before exploring responses, we examine the normalized performance for the male and female athletes surveyed. The distributions are approximately normal in all cases, but the snatch and clean and jerk events have a wider distribution. This is likely the result of the importance of technique in these events, which leads to wider variation in addition to natural differences in strength.


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


![lift data](\assets\img\crossfit_explore\output_26_0.png)
    



```python
#total lift 
sns.histplot(data = df, x = 'total_lift', hue='gender',kde = True);
plt.title('Normalized total lift')
plt.xlabel(xlabel);
```


![total lift data](\assets\img\crossfit_explore\output_27_0.png)    

<h2><br></h2>
<h2>Athlete demographics and performance</h2>

Athlete demographics were collected as part of provided survey:
- Gender 
- BMI (from height and weight)
- Age
- Region

We explore the data and look for correlations with performance.

### Gender

Male competitors outnumber female competitors by 2.6 to 1. This is unsurprising as CrossFit is a strength-based sport, which typically attracts men. 


```python
sns.histplot(data = df, x = 'gender', hue = 'gender', hue_order = ['Female','Male'])
#df.groupby('gender').agg('count') #20996/7999
```

 ![gender lift histogram](\assets\img\crossfit_explore\output_29_1.png)   
    


We see expected gender differences in performance between the male and female groups. Male athletes outlifted female athletes on a body weight normalized basis by 24.5%. 


```python
sns.boxplot(x=df.gender, y=df.total_lift)
plt.ylabel('Total lift (bodyweight)')

#df[['total_lift','gender']].groupby('gender').agg('mean') #(6.05-4.86)/4.86
```

![mean age lifts](\assets\img\crossfit_explore\output_31_1.png) 
    


### BMI

The distributions of heights and weights of CrossFit competitors are not smooth despite the large sample size. However, converting these non-independent metrics to BMI results in normal distributions with a positive skew that is particularly pronounced in the male population. 


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
sns.histplot(data=df, x='height', hue='gender', binwidth=1,kde=True, hue_order=['Female','Male'])
plt.subplot(1,2,2)
sns.histplot(data=df, x='weight', hue='gender', binwidth=5, kde=True, hue_order = ['Female','Male'])
plt.show()
```


![gender](\assets\img\crossfit_explore\output_34_0.png)    
    



```python
sns.histplot(data = df, x='BMI',hue='gender',binwidth=1,kde=True, hue_order=['Female','Male'])
plt.show()
```


![BMI](\assets\img\crossfit_explore\output_35_0.png)    
    


The figures below show BMI and total lift unaggregated in the left plot and aggregated to the nearest BMI with standard error in the right plot. Male athletes tend towards higher BMIs and lift more weight as expected. Interestingly, there appears to be a strong dependency between BMI and total lift with performance peaks observed at BMIs of 23 kg/m2 and 27 kg/m2 for female and male athletes respectively. Likely, this represents athletes gaining more muscle and eventually more fat, as more muscle would help performance and more fat would adversely affect performance. Thus, this analysis suggests that male and female athletes should adjust their training and diet to reach BMIs of 27 and 23 respectively with lean body compositions.


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
   


![agg BMI data](\assets\img\crossfit_explore\output_37_1.png) 
    


### Age 

The age distributions for male and female athletes is approximately normal with a slight positive skew that is more pronounced for female athletes. The mean ages for both male and female populations was 32. 


```python
sns.histplot(data = df, x = 'age', hue='gender',binwidth=1,kde = True)
plt.show()
#df[['gender','age']].groupby('gender').agg('mean')
```


 ![age data](\assets\img\crossfit_explore\output_39_0.png)   
    


The normalized total lift for all athletes is shown against age in the figure on the left below. Male competitors lift slightly more, and a general decrease in performance past 25 is apparent. This data is grouped to the nearest BMI and shown with standard error on the plot on the right. Male performance shows a clear peak around 20 years of age, and female performance maintains a plateau until around 30 years of age. Both populations show distinct decreases in performance with increased age.

Interestingly, male performance appears to be more sensitive to age than female performance. Testosterone is extremely important to building and maintaining muscle and strength, and it is possible that declines in testosterone characteristic to male aging underly this observation. Slower recovery with age in both genders is likely also a significant factor of age-related decline. 


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


![age data](\assets\img\crossfit_explore\output_41_1.png)
    


### Region

Most competitors are based in the US, and CrossFit appears to be particularly popular in California. Outside the US, the most popular locations for CrossFit are Canada and Europe.


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


![region](\assets\img\crossfit_explore\output_43_0.png)   


CrossFit athletes in the US appear to slightly outperform CrossFit athletes outside of the US. This could be due to increased local competition driving athletes to higher levels of performance or access to sophisticated training facilities and experienced coaches. 


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


![region](\assets\img\crossfit_explore\output_45_0.png)    
    

<h2><br></h2>
<h2>Lifestyle factors</h2>

Training and lifestyle choices are important factors for athlete performance. The following factors were surveyed:

- CrossFit age (time doing CrossFit)
- Two-a-day training session frequency
- Frequency of rest days 
- Athletic history 
- Eating habits

We examine these responses and identify key factors for improved performance. Histograms of these survey responses are shown below. 



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
    
![lifestyle data](\assets\img\crossfit_explore\output_41_1.png)    


Key take-aways: 

- Most CrossFit athletes wait to participate in the CrossFit games until they have some experience with the sport under their belt. 

- It is unusual for athletes to do CrossFit for more than four years, possibly due to the intensity of the training sessions in terms of time commitment and potential for injury. 

- Most athletes adhere to the recommended 1 day per week rest day schedule. 

- Most athletes have some athletic background prior to starting CrossFit. 

Next, we look at how these factors correlate to performance. 


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


![lifestyle data](\assets\img\crossfit_explore\output_49_0.png)    
    


CrossFit age (years doing CrossFit), frequency of two-a-day session, taking sufficient rest days, and athletic background all correlate with improved athletic performance as would be expected from conventional knowledge. Interestingly, starting CrossFit alone rather than at a facility with a coach appears to also correlate with improved performance. This is perhaps indirectly a result of self-starting athletes having a stronger athletic history. Reported eating habits, however, do not appear strongly correlated with performance. 

<h2><br></h2>
<h2>Has CrossFit been life changing?</h2>

Previous analysis focused on athlete attributes and lifestyle. Survey respondents were additionally asked if CrossFit has been life changing, and this reponse can be used to isolate and guage the effect of attitude on performance. 


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


![life changing](\assets\img\crossfit_explore\output_52_0.png)    
    


Interestingly, strong positive attitudes towards CrossFit do not appear to correlate with improved performance. This is counter intuitive as enjoying the ritual and community of CrossFit should increase the likelihood of continuing to train and participate in CrossFit.

<h2><br>/h2<>
<h2>Feature correlation</h2>

The Pearson correlation coefficient is a statistical measure that quantifies the strength and direction of the linear relationship between two continuous variables. It ranges from -1 to 1, where a value of 1 indicates a perfect positive correlation, -1 indicates a perfect negative correlation, and 0 indicates no linear correlation. Frequency of two-a-day training sessions, completing a level 1 certification, starting CrossFit alone, and working as a CrossFit trainer are all correlated to improved performance, while having no athletic background prior to CrossFit is correlated to worse performance. 

The strong non-linear correlations between performance and BMI and age are not captured in the above matrix. It is important to note that the Pearson correlation coefficient only measures linear relationships and may not capture non-linear associations or other types of relationships between variables.


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
    
![correlation matrix](\assets\img\crossfit_explore\output_55_1.png)


```python
#pickeling cleaned data
df.to_pickle('cleaned_cf_data.pkl')
```
<h2><br></h2>
<h2>Conclusion and recommendations</h2>

The results of this investigation indicate that key factors for CrossFit athlete performance are body composition, age, and gender. Other contributing factors include CrossFit age (years doing CrossFit), frequency of two-a-day sessions, taking sufficient rest days, and athletic background. Interestingly, passion for CrossFit and dietary choices did not correlate with athlete performance. 

CrossFit athletes are encouraged to follow a paleo diet. A modern paleo diet includes fruits, vegetables, lean meats, eggs, nuts and seeds and is not calorie or macro based. However, adherence to a paleo diet is uncorrelated with athlete performance. To better understand how athlete diet influences performance, additional questions could be added to future surveys. For example, caloric intake and protein consumption may be better predictors of performance, as sufficient protein and fuel are required to build and maintain muscle. If these features can be correlated with improved performance, CrossFit as an organization may have sufficient grounds to restructure their dietary recommendations for athletes. 

Finally, this data indicates that the key factors for athlete performance are largely predetermined by age and gender. However, there are key take aways for athletes seeking to improve their performance from their baseline: 
- Athletes should target lean body compositions and BMIs of 23 kg/m<sup>2</sup> and 27 mg/m<sup>

2</sup> for females and males respectively.

- Training should be managed for long-term adherence to CrossFit. While taking an overly aggressive approach to training may seem advantageous, the data indicates that there is a high risk of burnout or injury after four years, while performance continues to improve with CrossFit experience. 

- Adding two-a-day training sessions is beneficial to performance with no apparent risk of overtraining. 

- Taking one rest day a week will improve performance without adverse effects. 

