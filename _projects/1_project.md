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