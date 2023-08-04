---
layout: page
title: Getting into SHAP
description: Genetic drivers of breast cancer mortality are extracted from a black-box model as SHAP values. 
img: assets/img/cancer_shap/SHAP_image.jpg
importance: 3
category: Scikit-learn
---
<h1>Getting into SHAP</h1>
<h4><i>Identifying drivers of breast cancer related mortality with SHAP values</i></h4>
<h3><br></h3>
<h2>Introduction</h2>
In this project we use SHAP values to assess the drivers of breast cancer mortality from the Molecular Taxonomy of Breast Cancer International Consortium, [(METABRIC) database](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric). The METABRIC database is a Canada-UK Project which contains targeted sequencing data of 1,980 primary breast cancer samples originally downloaded from [cBioPortal](https://www.cbioportal.org/) and available on Kaggle.  A full understanding of the genetic underpinnings of cancer would enable scientists to develop effective treatments for each target and clinicians to select optimal treatment strategies. A full exploration of this dataset and development of predictive model can be found in [Extracting genetic causes of breast cancer mortality from binary classifiers](https://ninacilliers.github.io/projects/breast_cancer/). We first present needed context for the interpretation of SHAP values.

<h3><br></h3>
<h3>What is game theory? </h3>
Game theory is a branch of mathematics and economics that deals with the study of strategic decision-making in situations where the outcome of one individual's choices depends on the actions of others. It provides a framework to analyze interactions between rational agents and the strategies they employ to achieve their objectives. In the context of machine learning and explainable AI, SHAP values draw inspiration from game theory principles to attribute contributions to each feature in a prediction, allowing for a more interpretable and transparent understanding of the model's decision-making process.
<h3><br></h3>

<h3>What is a SHAP  value?</h3>
A SHAP value (SHapley Additive exPlanations) is a concept from cooperative game theory that has been adapted for use in machine learning and explainable artificial intelligence (AI). The idea is to attribute the prediction outcome to the individual features by considering all possible feature combinations and calculating their respective contributions. These contributions are then combined to provide a comprehensive and coherent explanation of the model's decision. This can explain predictions made by complex models, such as machine learning models. SHAP values can be positive or negative, depending on the relationship between the target and the feature. The magnitude of the SHAP value indicates the contribution of the feature.
<h3><br></h3>

<h3>How are SHAP values used?</h3>

- <b> Model validation </b> - Understanding how a model works increases user trust.
<br>
- <b>Debugging</b> - Interpretability allows for easier identification of errors.
<br>
- <b>Bias reduction</b> - Models that are used to make real-world decisions can be evaluated for bias. 
<br>
- <b>Scientific understanding</b> - Key features can be identified for further exploration by 
scientists and researchers.
<br>
- <b>Local interpretability</b> - Factors influencing individual predictions can be examined.
<h3><br></h3>

<h3>What conditions must SHAP values satisfy?</h3>
The SHAP value applies primarily in situations when the contributions of each actor are unequal, but they work in cooperation with each other to obtain the payoff.

- All the gains from cooperation are distributed among the players—none is wasted.
- Players that make equal contributions receive equal payoffs.
- The game cannot be divided into a set of smaller games that together achieve greater total gains
- A player that makes zero marginal contribution to the gains from cooperation receives zero payoff.

<h3><br></h3>
    
<h3>How do SHAP values differ from permutation importances?</h3>
Permutation importance is based on the decrease in model performance after scrambling the values of a feature, while SHAP importance is based on the magnitude of feature attributions. To assess feature importance by permutation, the values of a single feature are shuffled and the resulting drop in model performance is taken to represent the feature importance. The greater the drop in performance, the more important the feature is considered to be. Permutation importance does not offer local interpretability and is often less consistent. Permutation importance is computationally less intensive as only one case is considered per feature importance, whereas the computing time for SHAP values grows exponentially with feature number.
<h3><br></h3>

<h3>How are SHAP values calculated?</h3>
The equation to compute the SHAP value of a single feature (A) is presented below with annotation, from [\[1\]](http://adamlineberry.ai/shap/). Here, the contribution of A is considered in a collation of features A,B, and C. Possible arrangements with A in the first, second and third position and shown. These permutations are grouped into four coalitions, each with a distinct functional value before A is added to the set. The difference in predicted value of the model after and before A is added to the coalition is weighted by the probability the given arrangement of A,B and C occurs and summed over all coalitions. <br><br>

![png](/assets/img/cancer_shap/shapley-equation.png)<br>

In the above equation, F is the set of all features, so in this case F={A,B,C}. S is a subset of features. A single dash '\\' denotes 'without.' The fractional perfector in the sum indicates the probability of a coalition, and the term in brackets represent A's contribution to the coalition.
<br><br>

<h3><br></h3>
<h3>What is the computational cost of SHAP value determination?</h3>
Computing a single SHAP value requires sampling many coalitions. The computation time for SHAP value determination grows exponentially with feature number using the equation shown above, which would make calculations intractable on datasets with many features. Luckily approximate methods and optimizations are available, such as the Kernel SHAP method in Python's SHAP package, which will be used here. Kernel SHAP uses a special weighted linear regression to compute the importance of each feature. The coefficients from this local linear regression are SHAP values. 

<h2><br></h2>
<h2>Project Outline</h2>

1. Feature importance from SHAP<br>
   - SHAP values are determined using the SHAP kernel method in the Python SHAP package. <br>
   - Local SHAP values for distinct patient outcomes are visualized. <br>
   - Global SNAP values are visualized in force and beeswarm plots. <br><br>
2. Comparison to permutation importance scores <br>
   - Feature importances from permutation and SHAP indicate that similar features underly breast cancer mortality. <br>
   - Additional features identified by SHAP are contextualized in the existing clinical breast cancer research space. 

<h2><br></h2>
<h2>Feature importance from SHAP</h2>
<h3><br></h3>
<h3>Calculating SHAP value using Kernel SHAP</h3>
<br>

<details>
  <summary>Click to expand hidden code. </summary>

  <pre>
  import pickle
  import shap
  import pandas as pd
  shap.initjs()
  </pre>

  <pre>
  #loading cleaned data from previous project
  _ = pickle.load(open('./test_train_cancer_data.pickle', 'rb'))
  X_train= _[0]
  y_train = _[1]
  X_test = _[2]
  y_test = _[3]
  </pre>

  <pre>
  #loading trained model from previous project
  model = pickle.load(open('./svc_cancer_model.pickle','rb'))
  </pre>

</details>


```python
#load or calculate SHAP values 
try: 
    with open('shap.pkl','rb') as f:
        shap_values = pickle.load(f)

except:
    explainer = shap.KernelExplainer(model.predict,X_test)
    shap_values = explainer(X_test)
    with open('shap.pkl','wb') as f:
        pickle.dump(shap_values,f)
    f.close()
```

<h3><br></h3>
<h3>Exploring local SHAP values</h3>
Local SHAP values are shown in waterfall plots for two cases. In the first case, the model predicts survival, and in the second case the model predicts mortality. Each plot starts at the average expected value of the model and shows the pathway to the predicted output. The value in grey next to the feature name indicates the local value of each feature. These plots could be generated for each patient to help doctors develop custom treatment plans for patients.


```python
#preidicted survival
shap.plots.waterfall(shap_values[0],max_display=15)
```


    
![png](/assets/img/cancer_SHAP/output_6_0.png)



```python
#preidicted mortality
shap.plots.waterfall(shap_values[1],max_display=15)
```


    
![png](/assets/img/cancer_SHAP/output_7_0.png)
    


<h3><br></h3>
<h3>Exploring global SHAP values</h3>

The force plot is a useful way to view all local SHAP values. In the above waterfall plots, the local SHAP value of features is spatially stratified. The same information can be condensed into a force plot, shown below for case 1 above. We see the path to the predicted value in one dimension with the input from every feature shown as a blue or red arrow:


```python
shap.plots.force(shap_values[0])
```
![png](/assets/img/cancer_SHAP/force_plot_local.png)

Individual force plots are combined into an interactive global plot by stacking all local force plots together and rotating vertically. Samples can be arranged by predicted model value or by any input feature on the x-axis, and the SHAP values of all features or individual features can be selected for visualization on the y-axis. Here we order the samples by age at diagnosis from youngest to oldest, and visualize all local SHAP values. We see the model shift from predicting survival to death along the x-axis:


```python
shap.plots.force(shap_values)
```

![png](/assets/img/cancer_SHAP/force_plot_all.png)


We also look at the average SHAP value of features by averaging the absolute value of local SHAP values for each feature. The directionality of each SHAP value is lost, but we gain an efficient summary of what is important to the model globally:


```python
shap.plots.bar(shap_values,max_display=20)
```

    
![png](/assets/img/cancer_SHAP/output_13_0.png)<br>
    


Using a beeswarm plot, we visualize global SHAP averages as in the above plot with directional and distributional information. Each local SHAP value is plotted a single point against the x-axis, and recurring SHAP values are stacked vertically for each feature. The resulting stacking and clustering of local SHAP values enables us to view the global distribution. The color of each point is set by the value of each individual feature, so that relatively high feature values are red and relatively low feature values are blue. Thus, we can visualize the functional relationship between feature and predicted output For example, we see that the probability of mortality increases with age but that the probability of  mortality decreases with bbc3 gene expression level.


```python
shap.plots.beeswarm(shap_values,max_display=20)
```
    

    
![png](/assets/img/cancer_SHAP/output_15_1.png)<br>
    


<h2><br></h2>
<h2>Comparison to permutation importance scores</h2>

Feature importances from permutation were previously determined in [Extracting genetic causes of breast cancer mortality from binary classifiers](), and a summarizing figure is shown below:

![img](/assets/img/cancer_SHAP/permutation_importance.png)

Several features are identified by both techniques: 
- Age at diagnosis 
- BBC3
- CDKN2A
- INHBA
- STAT5A
- CHEK2
- SIK1
- ALK

Other features are only identified by SHAP:
- Positive lymph nodes
- Tumor size
- KDR
- WFDC5
- DNAH2
- RAD50
- NCOA3
- SMAD2
- HRAS
- SMARCD1

And others are only identified by permutation:
- cohort 1 and 3
- Tumor stage
- Surgery type (mastectomy)
- CASP8
- JAK1
- NOTCH3
- NRIP1
- CDKN2C
- E2F6
- GATA

Correlations between both positive lymph nodes and tumor size and mortality were identified previously during EDA, and the emergence of these features as significant by SHAP value analysis is not surprising. The genes of importance to both models and permutations were [previously explored](https://ninacilliers.github.io/projects/breast_cancer/). The top 5 genes identified by SHAP value analysis are explored below: 

1. **KDR**  - Kinase insert domain receptor (KDR) is the primary vascular endothelial growth factor receptor mediating survival, growth, and migration of endothelial cells and is expressed in tumor cells. Favorable prognosis has been observed when KDR is highly expressed, and expression levels of another gene ANLN are low [\[2\]](https://doi.org/10.3389/fgene.2019.00790) <br><br>
2. **WFDC5** - The role of WFDC5 expression in breast cancer has not been extensively investigated. The WFDC5 protein is a member of the whey acidic protein (WAP) domain with a four-disulfide core, which is characteristic of protease inhibition. WFDC5 has been shown to be upregulated in genes unerong induced apoptosis [\[3\]]( https://doi.org/10.1006/bbrc.1999.1123).<br><br>
3. **DNAH2** - DNAH2 is a member of the dynein axonemal heavy chain (DNAH) family of genes involved in cell motility, and mutations are frequently reported in malignant tumors and may increase the efficacy of chemotherapy [\[4\]](https://doi.org/10.1186/s12967-019-1867-6). The relationship between DNAH2 and breast cancer has not been extensively studied. <br><br>
5. **RAD50** - The RAD50 protein is essential for cell growth and viability and is involved in DNA double-strand break repair. RAD50 germline mutations are associated with poor survival in BRCA1/2-negative breast cancer patients [\[5\]](https://doi.org/10.1002/ijc.31579).<br>

<h2><br></h2>
<h2>Works referenced </h2>

1. Getting Up to Speed with SHAP for Model Interpretability. Adam Lindeberry, Machine Learning Blog, 2022. http://adamlineberry.ai/shap/.
2. Dai, X., Mei, Y., Chen, X., & Cai, D. (2018). ANLN and KDR Are Jointly Prognostic of Breast Cancer Survival and Can Be Modulated for Triple Negative Breast Cancer Control. Frontiers in Genetics, 10.
3. Horikoshi, N., Cong, J., Kley, N., & Shenk, T. (1999). Isolation of Differentially Expressed cDNAs from p53-Dependent Apoptotic Cells: Activation of the Human Homologue of the Drosophila Peroxidasin Gene. Biochemical and Biophysical Research Communications, 261(3), 864-869.
4. Zhu, C., Yang, Q., Xu, J. et al. Somatic mutation of DNAH genes implicated higher chemotherapy response rate in gastric adenocarcinoma patients. J Transl Med 17, 109 (2019).
5. Fan, C., Zhang, J., Ouyang, T., Li, J., Wang, T., Fan, Z., Fan, T., Lin, B., & Xie, Y. (2018). RAD50 germline mutations are associated with poor survival in BRCA1/2-negative breast cancer patients. International journal of cancer, 143(8), 1935–1942.
