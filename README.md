# 123OFAI_California_Linear_Regressor_PCA
Projects during the 123OfAI AlphaML Course - Implement a Linear Regressor on the California Housing Dataset: Dimensionality Reduction

IMPORTING MODULES


```python
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

```


```python
# get house information
houses = fetch_california_housing()

x = houses.data
y = houses.target

df_data = pd.DataFrame(houses.data, columns=houses.feature_names)
df_target = pd.DataFrame(houses.target, columns=['Target'])
df_total = pd.concat([df_data, df_target], axis=1, sort=False)

df_data.head()
df_target.head()

print(type(houses.data))
print(type(df_data))
print('the number of rows and colums are ' + str(df_data.shape))
print('the number of rows and colums are ' + str(df_target.shape))
print('the number of rows and colums are ' + str(df_total.shape))

print('\nthe columns in the data are - \n')
[print('\t* ', i) for i in df_data.columns.values]
print('\nthe columns in the target are - \n')
[print('\t* ', i) for i in df_target.columns.values]
print('\nthe columns in the total data are - \n')
[print('\t* ', i) for i in df_total.columns.values]
```

    <class 'numpy.ndarray'>
    <class 'pandas.core.frame.DataFrame'>
    the number of rows and colums are (20640, 8)
    the number of rows and colums are (20640, 1)
    the number of rows and colums are (20640, 9)
    
    the columns in the data are - 
    
    	*  MedInc
    	*  HouseAge
    	*  AveRooms
    	*  AveBedrms
    	*  Population
    	*  AveOccup
    	*  Latitude
    	*  Longitude
    
    the columns in the target are - 
    
    	*  Target
    
    the columns in the total data are - 
    
    	*  MedInc
    	*  HouseAge
    	*  AveRooms
    	*  AveBedrms
    	*  Population
    	*  AveOccup
    	*  Latitude
    	*  Longitude
    	*  Target
    




    [None, None, None, None, None, None, None, None, None]




```python
# Data visualization

"""It gives an overall understanding of the nature of the data as well as the operations/tranformations that the models to train to could benefit from."""
```




    'It gives an overall understanding of the nature of the data as well as the operations/tranformations that the models to train to could benefit from.'




```python
df_total.hist(bins=100, figsize=(20,15))
```




    array([[<Axes: title={'center': 'MedInc'}>,
            <Axes: title={'center': 'HouseAge'}>,
            <Axes: title={'center': 'AveRooms'}>],
           [<Axes: title={'center': 'AveBedrms'}>,
            <Axes: title={'center': 'Population'}>,
            <Axes: title={'center': 'AveOccup'}>],
           [<Axes: title={'center': 'Latitude'}>,
            <Axes: title={'center': 'Longitude'}>,
            <Axes: title={'center': 'Target'}>]], dtype=object)




    
![png](California_dataset_Linear_Regression_PCA_files/California_dataset_Linear_Regression_PCA_4_1.png)
    



```python
"""From the plots it is clear that:

The attributes have different scales so rescaling is needed
The targetand median have clear outliers. Those will need to be deleted
I suppose that the house age have a peak as a result of the fact that houses over a certain age end up getting that maximum number"""
```




    'From the plots it is clear that:\n\nThe attributes have different scales so rescaling is needed\nThe targetand median have clear outliers. Those will need to be deleted\nI suppose that the house age have a peak as a result of the fact that houses over a certain age end up getting that maximum number'




```python
# Data cleaning
""" Not giving any model precision increase!"""

# df_total[df_total['Target']>=5]['Target'].value_counts().head()
# df_total[df_total['MedInc']>=15]['MedInc'].value_counts().head()
# df_total=df_total.loc[df_total['Target']<5,:]
# df_total=df_total.loc[df_total['Target']<15,:]
```




    ' Not giving any model precision increase!'




```python
# Feature engineering
""" Not giving any precition increase either """

#df_total['r/b']=df_total['AveRooms']/df_total['AveBedrms']
```




    ' Not giving any precition increase either '




```python
# Standardize the data
"""
Many estimators are designed with the assumption that each feature takes values close to zero or more importantly that all features vary on comparable scales. In particular, metric-based and gradient-based estimators often assume approximately standardized data (centered features with unit variances). A notable exception are decision tree-based estimators that are robust to arbitrary scaling of the data.

A Box Cox transformation is a way to transform non-normal dependent variables into a normal shape.

PCA is largely affected by scales as well. Here we have hence decided to use sklearn’s StandardScaler in order to scale the data to zero mean and unit variance.

SOURCES

https://stats.stackexchange.com/questions/105592/not-normalizing-data-before-pca-gives-better-explained-variance-ratio

https://ostwalprasad.github.io/machine-learning/PCA-using-python.html

https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html

"""

```




    '\nMany estimators are designed with the assumption that each feature takes values close to zero or more importantly that all features vary on comparable scales. In particular, metric-based and gradient-based estimators often assume approximately standardized data (centered features with unit variances). A notable exception are decision tree-based estimators that are robust to arbitrary scaling of the data.\n\nA Box Cox transformation is a way to transform non-normal dependent variables into a normal shape.\n\nPCA is largely affected by scales as well. Here we have hence decided to use sklearn’s StandardScaler in order to scale the data to zero mean and unit variance.\n\nSOURCES\n\nhttps://stats.stackexchange.com/questions/105592/not-normalizing-data-before-pca-gives-better-explained-variance-ratio\n\nhttps://ostwalprasad.github.io/machine-learning/PCA-using-python.html\n\nhttps://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html\n\n'




```python
from sklearn.preprocessing import StandardScaler

def transform1(X):
  x = StandardScaler().fit_transform(X)
  X = pd.DataFrame(x, columns=X.columns)
  return X

def transform2(X,y=None):
    import numpy as np
    from scipy.special import boxcox1p        
    X['AveRooms']=X['AveRooms'].apply(lambda x: boxcox1p(x,0.25))
    X['AveBedrms']=X['AveBedrms'].apply(lambda x: boxcox1p(x,0.25))
    X['HouseAge']=X['HouseAge'].apply(lambda x: boxcox1p(x,0.25))
    X['Population']=X['Population'].apply(lambda x: boxcox1p(x,0.25))
    X['AveOccup']=X['AveOccup'].apply(lambda x: boxcox1p(x,0.25))
    X['Latitude']=X['Latitude'].apply(lambda x: boxcox1p(x,0.25))
    X['MedInc']=X['MedInc'].apply(lambda x: boxcox1p(x,0.25))
    # an offset is needed becouse the data is negative
    X['Longitude']=X['Longitude'].apply(lambda x: boxcox1p(x+125,0.25))
    X['Target']=X['Target'].apply(lambda x: boxcox1p(x,0.25))
    return X

    
df_total = transform2(df_total)
```


```python
df_total.hist(bins=100, figsize=(20,15))
```




    array([[<Axes: title={'center': 'MedInc'}>,
            <Axes: title={'center': 'HouseAge'}>,
            <Axes: title={'center': 'AveRooms'}>],
           [<Axes: title={'center': 'AveBedrms'}>,
            <Axes: title={'center': 'Population'}>,
            <Axes: title={'center': 'AveOccup'}>],
           [<Axes: title={'center': 'Latitude'}>,
            <Axes: title={'center': 'Longitude'}>,
            <Axes: title={'center': 'Target'}>]], dtype=object)




    
![png](California_dataset_Linear_Regression_PCA_files/California_dataset_Linear_Regression_PCA_10_1.png)
    



```python
# Correlation analysis
```


```python
corr = df_total.corr()
plt.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=60);
plt.yticks(range(len(corr.columns)), corr.columns);
plt.colorbar()
plt.show()
df_total.corr().style.background_gradient(cmap='coolwarm')
```


    
![png](California_dataset_Linear_Regression_PCA_files/California_dataset_Linear_Regression_PCA_12_0.png)
    





<style type="text/css">
#T_1a2a2_row0_col0, #T_1a2a2_row1_col1, #T_1a2a2_row2_col2, #T_1a2a2_row3_col3, #T_1a2a2_row4_col4, #T_1a2a2_row5_col5, #T_1a2a2_row6_col6, #T_1a2a2_row7_col7, #T_1a2a2_row8_col8 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_1a2a2_row0_col1, #T_1a2a2_row3_col8 {
  background-color: #5977e3;
  color: #f1f1f1;
}
#T_1a2a2_row0_col2 {
  background-color: #f3c8b2;
  color: #000000;
}
#T_1a2a2_row0_col3 {
  background-color: #455cce;
  color: #f1f1f1;
}
#T_1a2a2_row0_col4 {
  background-color: #85a8fc;
  color: #f1f1f1;
}
#T_1a2a2_row0_col5, #T_1a2a2_row7_col1 {
  background-color: #6687ed;
  color: #f1f1f1;
}
#T_1a2a2_row0_col6 {
  background-color: #cbd8ee;
  color: #000000;
}
#T_1a2a2_row0_col7 {
  background-color: #d5dbe5;
  color: #000000;
}
#T_1a2a2_row0_col8 {
  background-color: #f59d7e;
  color: #000000;
}
#T_1a2a2_row1_col0, #T_1a2a2_row1_col2, #T_1a2a2_row1_col4, #T_1a2a2_row4_col1, #T_1a2a2_row4_col3, #T_1a2a2_row5_col8, #T_1a2a2_row6_col7, #T_1a2a2_row7_col6, #T_1a2a2_row8_col5 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_1a2a2_row1_col3 {
  background-color: #4358cb;
  color: #f1f1f1;
}
#T_1a2a2_row1_col5 {
  background-color: #7396f5;
  color: #f1f1f1;
}
#T_1a2a2_row1_col6 {
  background-color: #d8dce2;
  color: #000000;
}
#T_1a2a2_row1_col7 {
  background-color: #c9d7f0;
  color: #000000;
}
#T_1a2a2_row1_col8 {
  background-color: #86a9fc;
  color: #f1f1f1;
}
#T_1a2a2_row2_col0 {
  background-color: #f0cdbb;
  color: #000000;
}
#T_1a2a2_row2_col1 {
  background-color: #465ecf;
  color: #f1f1f1;
}
#T_1a2a2_row2_col3 {
  background-color: #f6bea4;
  color: #000000;
}
#T_1a2a2_row2_col4 {
  background-color: #5d7ce6;
  color: #f1f1f1;
}
#T_1a2a2_row2_col5, #T_1a2a2_row4_col0, #T_1a2a2_row7_col3 {
  background-color: #688aef;
  color: #f1f1f1;
}
#T_1a2a2_row2_col6 {
  background-color: #ead5c9;
  color: #000000;
}
#T_1a2a2_row2_col7 {
  background-color: #cedaeb;
  color: #000000;
}
#T_1a2a2_row2_col8 {
  background-color: #b1cbfc;
  color: #000000;
}
#T_1a2a2_row3_col0, #T_1a2a2_row6_col8 {
  background-color: #445acc;
  color: #f1f1f1;
}
#T_1a2a2_row3_col1 {
  background-color: #5f7fe8;
  color: #f1f1f1;
}
#T_1a2a2_row3_col2 {
  background-color: #f7b89c;
  color: #000000;
}
#T_1a2a2_row3_col4, #T_1a2a2_row5_col0 {
  background-color: #5875e1;
  color: #f1f1f1;
}
#T_1a2a2_row3_col5, #T_1a2a2_row6_col4 {
  background-color: #5b7ae5;
  color: #f1f1f1;
}
#T_1a2a2_row3_col6 {
  background-color: #e2dad5;
  color: #000000;
}
#T_1a2a2_row3_col7 {
  background-color: #d9dce1;
  color: #000000;
}
#T_1a2a2_row4_col2 {
  background-color: #506bda;
  color: #f1f1f1;
}
#T_1a2a2_row4_col5 {
  background-color: #a7c5fe;
  color: #000000;
}
#T_1a2a2_row4_col6 {
  background-color: #c3d5f4;
  color: #000000;
}
#T_1a2a2_row4_col7 {
  background-color: #e7d7ce;
  color: #000000;
}
#T_1a2a2_row4_col8 {
  background-color: #6f92f3;
  color: #f1f1f1;
}
#T_1a2a2_row5_col1 {
  background-color: #84a7fc;
  color: #f1f1f1;
}
#T_1a2a2_row5_col2 {
  background-color: #6c8ff1;
  color: #f1f1f1;
}
#T_1a2a2_row5_col3 {
  background-color: #4e68d8;
  color: #f1f1f1;
}
#T_1a2a2_row5_col4, #T_1a2a2_row8_col2 {
  background-color: #b3cdfb;
  color: #000000;
}
#T_1a2a2_row5_col6 {
  background-color: #c7d7f0;
  color: #000000;
}
#T_1a2a2_row5_col7 {
  background-color: #e8d6cc;
  color: #000000;
}
#T_1a2a2_row6_col0 {
  background-color: #4b64d5;
  color: #f1f1f1;
}
#T_1a2a2_row6_col1 {
  background-color: #82a6fb;
  color: #f1f1f1;
}
#T_1a2a2_row6_col2 {
  background-color: #9dbdff;
  color: #000000;
}
#T_1a2a2_row6_col3 {
  background-color: #7da0f9;
  color: #f1f1f1;
}
#T_1a2a2_row6_col5 {
  background-color: #536edd;
  color: #f1f1f1;
}
#T_1a2a2_row7_col0 {
  background-color: #5e7de7;
  color: #f1f1f1;
}
#T_1a2a2_row7_col2 {
  background-color: #6384eb;
  color: #f1f1f1;
}
#T_1a2a2_row7_col4 {
  background-color: #a1c0ff;
  color: #000000;
}
#T_1a2a2_row7_col5, #T_1a2a2_row8_col1 {
  background-color: #96b7ff;
  color: #000000;
}
#T_1a2a2_row7_col8 {
  background-color: #6788ee;
  color: #f1f1f1;
}
#T_1a2a2_row8_col0 {
  background-color: #f6a283;
  color: #000000;
}
#T_1a2a2_row8_col3 {
  background-color: #4c66d6;
  color: #f1f1f1;
}
#T_1a2a2_row8_col4 {
  background-color: #80a3fa;
  color: #f1f1f1;
}
#T_1a2a2_row8_col6 {
  background-color: #bed2f6;
  color: #000000;
}
#T_1a2a2_row8_col7 {
  background-color: #d2dbe8;
  color: #000000;
}
</style>
<table id="T_1a2a2">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_1a2a2_level0_col0" class="col_heading level0 col0" >MedInc</th>
      <th id="T_1a2a2_level0_col1" class="col_heading level0 col1" >HouseAge</th>
      <th id="T_1a2a2_level0_col2" class="col_heading level0 col2" >AveRooms</th>
      <th id="T_1a2a2_level0_col3" class="col_heading level0 col3" >AveBedrms</th>
      <th id="T_1a2a2_level0_col4" class="col_heading level0 col4" >Population</th>
      <th id="T_1a2a2_level0_col5" class="col_heading level0 col5" >AveOccup</th>
      <th id="T_1a2a2_level0_col6" class="col_heading level0 col6" >Latitude</th>
      <th id="T_1a2a2_level0_col7" class="col_heading level0 col7" >Longitude</th>
      <th id="T_1a2a2_level0_col8" class="col_heading level0 col8" >Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_1a2a2_level0_row0" class="row_heading level0 row0" >MedInc</th>
      <td id="T_1a2a2_row0_col0" class="data row0 col0" >1.000000</td>
      <td id="T_1a2a2_row0_col1" class="data row0 col1" >-0.146516</td>
      <td id="T_1a2a2_row0_col2" class="data row0 col2" >0.527787</td>
      <td id="T_1a2a2_row0_col3" class="data row0 col3" >-0.110333</td>
      <td id="T_1a2a2_row0_col4" class="data row0 col4" >0.019547</td>
      <td id="T_1a2a2_row0_col5" class="data row0 col5" >-0.036270</td>
      <td id="T_1a2a2_row0_col6" class="data row0 col6" >-0.083011</td>
      <td id="T_1a2a2_row0_col7" class="data row0 col7" >-0.013237</td>
      <td id="T_1a2a2_row0_col8" class="data row0 col8" >0.688867</td>
    </tr>
    <tr>
      <th id="T_1a2a2_level0_row1" class="row_heading level0 row1" >HouseAge</th>
      <td id="T_1a2a2_row1_col0" class="data row1 col0" >-0.146516</td>
      <td id="T_1a2a2_row1_col1" class="data row1 col1" >1.000000</td>
      <td id="T_1a2a2_row1_col2" class="data row1 col2" >-0.219691</td>
      <td id="T_1a2a2_row1_col3" class="data row1 col3" >-0.119871</td>
      <td id="T_1a2a2_row1_col4" class="data row1 col4" >-0.273806</td>
      <td id="T_1a2a2_row1_col5" class="data row1 col5" >0.010693</td>
      <td id="T_1a2a2_row1_col6" class="data row1 col6" >0.005362</td>
      <td id="T_1a2a2_row1_col7" class="data row1 col7" >-0.095173</td>
      <td id="T_1a2a2_row1_col8" class="data row1 col8" >0.075256</td>
    </tr>
    <tr>
      <th id="T_1a2a2_level0_row2" class="row_heading level0 row2" >AveRooms</th>
      <td id="T_1a2a2_row2_col0" class="data row2 col0" >0.527787</td>
      <td id="T_1a2a2_row2_col1" class="data row2 col1" >-0.219691</td>
      <td id="T_1a2a2_row2_col2" class="data row2 col2" >1.000000</td>
      <td id="T_1a2a2_row2_col3" class="data row2 col3" >0.591297</td>
      <td id="T_1a2a2_row2_col4" class="data row2 col4" >-0.131645</td>
      <td id="T_1a2a2_row2_col5" class="data row2 col5" >-0.027987</td>
      <td id="T_1a2a2_row2_col6" class="data row2 col6" >0.138555</td>
      <td id="T_1a2a2_row2_col7" class="data row2 col7" >-0.058171</td>
      <td id="T_1a2a2_row2_col8" class="data row2 col8" >0.219160</td>
    </tr>
    <tr>
      <th id="T_1a2a2_level0_row3" class="row_heading level0 row3" >AveBedrms</th>
      <td id="T_1a2a2_row3_col0" class="data row3 col0" >-0.110333</td>
      <td id="T_1a2a2_row3_col1" class="data row3 col1" >-0.119871</td>
      <td id="T_1a2a2_row3_col2" class="data row3 col2" >0.591297</td>
      <td id="T_1a2a2_row3_col3" class="data row3 col3" >1.000000</td>
      <td id="T_1a2a2_row3_col4" class="data row3 col4" >-0.151471</td>
      <td id="T_1a2a2_row3_col5" class="data row3 col5" >-0.076174</td>
      <td id="T_1a2a2_row3_col6" class="data row3 col6" >0.083644</td>
      <td id="T_1a2a2_row3_col7" class="data row3 col7" >0.015681</td>
      <td id="T_1a2a2_row3_col8" class="data row3 col8" >-0.083908</td>
    </tr>
    <tr>
      <th id="T_1a2a2_level0_row4" class="row_heading level0 row4" >Population</th>
      <td id="T_1a2a2_row4_col0" class="data row4 col0" >0.019547</td>
      <td id="T_1a2a2_row4_col1" class="data row4 col1" >-0.273806</td>
      <td id="T_1a2a2_row4_col2" class="data row4 col2" >-0.131645</td>
      <td id="T_1a2a2_row4_col3" class="data row4 col3" >-0.151471</td>
      <td id="T_1a2a2_row4_col4" class="data row4 col4" >1.000000</td>
      <td id="T_1a2a2_row4_col5" class="data row4 col5" >0.187266</td>
      <td id="T_1a2a2_row4_col6" class="data row4 col6" >-0.136856</td>
      <td id="T_1a2a2_row4_col7" class="data row4 col7" >0.116194</td>
      <td id="T_1a2a2_row4_col8" class="data row4 col8" >-0.003522</td>
    </tr>
    <tr>
      <th id="T_1a2a2_level0_row5" class="row_heading level0 row5" >AveOccup</th>
      <td id="T_1a2a2_row5_col0" class="data row5 col0" >-0.036270</td>
      <td id="T_1a2a2_row5_col1" class="data row5 col1" >0.010693</td>
      <td id="T_1a2a2_row5_col2" class="data row5 col2" >-0.027987</td>
      <td id="T_1a2a2_row5_col3" class="data row5 col3" >-0.076174</td>
      <td id="T_1a2a2_row5_col4" class="data row5 col4" >0.187266</td>
      <td id="T_1a2a2_row5_col5" class="data row5 col5" >1.000000</td>
      <td id="T_1a2a2_row5_col6" class="data row5 col6" >-0.109058</td>
      <td id="T_1a2a2_row5_col7" class="data row5 col7" >0.127576</td>
      <td id="T_1a2a2_row5_col8" class="data row5 col8" >-0.204878</td>
    </tr>
    <tr>
      <th id="T_1a2a2_level0_row6" class="row_heading level0 row6" >Latitude</th>
      <td id="T_1a2a2_row6_col0" class="data row6 col0" >-0.083011</td>
      <td id="T_1a2a2_row6_col1" class="data row6 col1" >0.005362</td>
      <td id="T_1a2a2_row6_col2" class="data row6 col2" >0.138555</td>
      <td id="T_1a2a2_row6_col3" class="data row6 col3" >0.083644</td>
      <td id="T_1a2a2_row6_col4" class="data row6 col4" >-0.136856</td>
      <td id="T_1a2a2_row6_col5" class="data row6 col5" >-0.109058</td>
      <td id="T_1a2a2_row6_col6" class="data row6 col6" >1.000000</td>
      <td id="T_1a2a2_row6_col7" class="data row6 col7" >-0.920040</td>
      <td id="T_1a2a2_row6_col8" class="data row6 col8" >-0.165399</td>
    </tr>
    <tr>
      <th id="T_1a2a2_level0_row7" class="row_heading level0 row7" >Longitude</th>
      <td id="T_1a2a2_row7_col0" class="data row7 col0" >-0.013237</td>
      <td id="T_1a2a2_row7_col1" class="data row7 col1" >-0.095173</td>
      <td id="T_1a2a2_row7_col2" class="data row7 col2" >-0.058171</td>
      <td id="T_1a2a2_row7_col3" class="data row7 col3" >0.015681</td>
      <td id="T_1a2a2_row7_col4" class="data row7 col4" >0.116194</td>
      <td id="T_1a2a2_row7_col5" class="data row7 col5" >0.127576</td>
      <td id="T_1a2a2_row7_col6" class="data row7 col6" >-0.920040</td>
      <td id="T_1a2a2_row7_col7" class="data row7 col7" >1.000000</td>
      <td id="T_1a2a2_row7_col8" class="data row7 col8" >-0.035070</td>
    </tr>
    <tr>
      <th id="T_1a2a2_level0_row8" class="row_heading level0 row8" >Target</th>
      <td id="T_1a2a2_row8_col0" class="data row8 col0" >0.688867</td>
      <td id="T_1a2a2_row8_col1" class="data row8 col1" >0.075256</td>
      <td id="T_1a2a2_row8_col2" class="data row8 col2" >0.219160</td>
      <td id="T_1a2a2_row8_col3" class="data row8 col3" >-0.083908</td>
      <td id="T_1a2a2_row8_col4" class="data row8 col4" >-0.003522</td>
      <td id="T_1a2a2_row8_col5" class="data row8 col5" >-0.204878</td>
      <td id="T_1a2a2_row8_col6" class="data row8 col6" >-0.165399</td>
      <td id="T_1a2a2_row8_col7" class="data row8 col7" >-0.035070</td>
      <td id="T_1a2a2_row8_col8" class="data row8 col8" >1.000000</td>
    </tr>
  </tbody>
</table>





```python
"""
It is clear from this plot that the average number of bedrooms and average number of rooms are somewhat correlated, so both variables move in the same direction by similar magnitude. This is also true for the median income and the average number of rooms. The longitude and magnitudes is strongly negatively correlated, which means that the variables move in opposite direction by the same magnitude.

The strongest correlation to the target is the median income for households within a block of houses (measured in tens of thousands of US Dollars). The rest of the signal shows barely any correlation to the target at all.
"""
```




    '\nIt is clear from this plot that the average number of bedrooms and average number of rooms are somewhat correlated, so both variables move in the same direction by similar magnitude. This is also true for the median income and the average number of rooms. The longitude and magnitudes is strongly negatively correlated, which means that the variables move in opposite direction by the same magnitude.\n\nThe strongest correlation to the target is the median income for households within a block of houses (measured in tens of thousands of US Dollars). The rest of the signal shows barely any correlation to the target at all.\n'




```python
# Split datat in train and test:
""" We do not apply PCA before the splitting as we would be leaking the information of test data. """
```




    ' We do not apply PCA before the splitting as we would be leaking the information of test data. '




```python
from sklearn.model_selection import train_test_split

df_target = df_total.loc[:, df_total.columns == 'Target']
df_data = df_total.loc[:, df_total.columns != 'Target']

xtrain,xtest,ytrain,ytest=train_test_split(df_data,df_target,test_size=0.1,random_state=0)

print('train data shape'+str(xtrain.shape))
print('test data shape'+str(xtest.shape))
print('train target shape'+str(ytrain.shape))
print('test target shape'+str(ytest.shape))
```

    train data shape(18576, 8)
    test data shape(2064, 8)
    train target shape(18576, 1)
    test target shape(2064, 1)
    


```python
# Checking the importance of features and plot them

"""As this dataset is not so big we can extract all the components and analyze the percentage of variance by these features (based on: https://stackoverflow.com/questions/50796024/feature-variable-importance-after-a-pca-analysis)"""
```




    'As this dataset is not so big we can extract all the components and analyze the percentage of variance by these features (based on: https://stackoverflow.com/questions/50796024/feature-variable-importance-after-a-pca-analysis)'




```python
from sklearn.decomposition import PCA

pcamodel = PCA(8)
pca = pcamodel.fit_transform(xtrain)
print(pca.shape)


# number of components
n_pcs= pcamodel.components_.shape[0]

# get the index of the most important feature of each component
most_important = [np.abs(pcamodel.components_[i]).argmax() for i in range(n_pcs)]

# extract the column names frmo the dataframe
initial_feature_names = df_data.columns.values

# get the names based on the principal component analysis
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

# pack the result into a dataframe
dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}
df = pd.DataFrame(dic.items())

print('variance ration of each feature:', pcamodel.explained_variance_ratio_)
print('features:', most_important_names)
print(df)
```

    (18576, 8)
    variance ration of each feature: [8.92625262e-01 6.32998236e-02 1.74674815e-02 1.56822277e-02
     5.54535165e-03 4.89672962e-03 3.75417011e-04 1.07706704e-04]
    features: ['Population', 'HouseAge', 'MedInc', 'Longitude', 'AveOccup', 'AveRooms', 'AveBedrms', 'Latitude']
         0           1
    0  PC0  Population
    1  PC1    HouseAge
    2  PC2      MedInc
    3  PC3   Longitude
    4  PC4    AveOccup
    5  PC5    AveRooms
    6  PC6   AveBedrms
    7  PC7    Latitude
    


```python
plt.plot(np.cumsum(pcamodel.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
```


    
![png](California_dataset_Linear_Regression_PCA_files/California_dataset_Linear_Regression_PCA_18_0.png)
    



```python
"""
We see that the five first components (PC0, PC1, PC2, PC3, PC4) account for approximately 90% of the variance. That would lead us to believe that using these 5 components, we would maintain most of the essential characteristics of the data.
"""
```




    '\nWe see that the five first components (PC0, PC1, PC2, PC3, PC4) account for approximately 90% of the variance. That would lead us to believe that using these 5 components, we would maintain most of the essential characteristics of the data.\n'




```python
# 3D Scatter plot of PCA1, PCA2 and PCA3:

import plotly.express as px

fig = px.scatter_3d(x=pca[:, 0],
                    y=pca[:, 1],
                    z=pca[:, 2])
fig.update_layout(
    title="PC1 vs PC2 vs PC3",
    scene = dict(xaxis = dict(title='PC1'),
                 yaxis = dict(title='PC2'),
                 zaxis = dict(title='PC3'),)
)

fig.show()
```




```python
# Effect of variables on each components

"""The components_ attribute provides principal axes in feature space, representing the directions of maximum variance in the data. This means, we can see influence on each of the components by features."""
```




    'The components_ attribute provides principal axes in feature space, representing the directions of maximum variance in the data. This means, we can see influence on each of the components by features.'




```python
import seaborn as sns

ax = sns.heatmap(pcamodel.components_,
                 cmap='YlGnBu',
                 yticklabels=["PCA"+str(x) for x in range(1,pcamodel.n_components_+1)],
                 xticklabels=list(xtrain.columns));
ax.set_aspect("equal")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
```




    [Text(0, 0.5, 'PCA1'),
     Text(0, 1.5, 'PCA2'),
     Text(0, 2.5, 'PCA3'),
     Text(0, 3.5, 'PCA4'),
     Text(0, 4.5, 'PCA5'),
     Text(0, 5.5, 'PCA6'),
     Text(0, 6.5, 'PCA7'),
     Text(0, 7.5, 'PCA8')]




    
![png](California_dataset_Linear_Regression_PCA_files/California_dataset_Linear_Regression_PCA_22_1.png)
    



```python
# Original data VS reduced data

""" Here we reduce the test data based on the PCA model trained by the training data. """
```




    ' Here we reduce the test data based on the PCA model trained by the training data. '




```python
# Compute the components and projected faces
pca = PCA(2).fit(xtrain)
xtraincomponents = pca.transform(xtrain)
xtestcomponents = pca.transform(xtest)
projected = pca.inverse_transform(xtraincomponents)

print("original shape:   ", xtrain.shape)
print("transformed train shape:", xtraincomponents.shape)
print("transformed test shape:", xtestcomponents.shape)

#print('\nthe columns in the data are - \n')
#[print('\t* ', i) for i in df_data.columns.values]
#print('\nthe columns in the target are - \n')
#[print('\t* ', i) for i in components.columns.values]
```

    original shape:    (18576, 8)
    transformed train shape: (18576, 2)
    transformed test shape: (2064, 2)
    


```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics


# We have created a function to print accuracy metrics which can be used
# to get accuracy metrics of all models in upcoming steps
def print_accuracy_report(y_test, y_pred, X_test, model):
    print('R Squared(Accuracy)', metrics.r2_score(y_test, y_pred))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)), '\n')


def LinearRegressionModel(xtrain, ytrain, xtest):
    regressor = LinearRegression()
    regressor.fit(xtrain, ytrain)
    ypred = regressor.predict(xtest)
    #for i, values in enumerate(y_test):
    #    print(str(y_pred[i]), str(y_test[i]))
    print_accuracy_report(ytest, ypred, xtest, regressor)
    return regressor


def RandomForestRegressorModel(xtrain, ytrain, xtest):
    rf = RandomForestRegressor(random_state=42)
    rf.fit(xtrain, ytrain)
    ypred = rf.predict(xtest)
    print(print_accuracy_report(ytest, ypred, xtest, rf))
    return rf


print('\t\t-- Linear regression results --')
print('\tWith original data')
LinearRegressionModel(xtrain, ytrain, xtest)
print('\tWith PCA data')
LinearRegressionModel(xtraincomponents, ytrain, xtestcomponents)

print('\t\t-- Random forest results --')
print('\tWith original data')
randomForestModel = RandomForestRegressorModel(xtrain, ytrain, xtest)
print('\tWith PCA data')
randomForestModel = RandomForestRegressorModel(xtraincomponents, ytrain, xtestcomponents)
```

    		-- Linear regression results --
    	With original data
    R Squared(Accuracy) 0.642378145744567
    Mean Absolute Error: 0.2100784221510826
    Mean Squared Error: 0.08142657835260234
    Root Mean Squared Error: 0.2853534270910415 
    
    	With PCA data
    R Squared(Accuracy) 0.0007301996364466046
    Mean Absolute Error: 0.38956022026508635
    Mean Squared Error: 0.2275227862237282
    Root Mean Squared Error: 0.47699348656321106 
    
    		-- Random forest results --
    	With original data
    

    c:\Users\SESA708153\Documents\123ofAI\123OFAI\Lib\site-packages\sklearn\base.py:1474: DataConversionWarning:
    
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
    
    

    R Squared(Accuracy) 0.8312081985267226
    Mean Absolute Error: 0.1329758169136136
    Mean Squared Error: 0.03843204402749926
    Root Mean Squared Error: 0.1960409243691206 
    
    None
    	With PCA data
    

    c:\Users\SESA708153\Documents\123ofAI\123OFAI\Lib\site-packages\sklearn\base.py:1474: DataConversionWarning:
    
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
    
    

    R Squared(Accuracy) -0.1438653253220099
    Mean Absolute Error: 0.4116709489098965
    Mean Squared Error: 0.2604456031667214
    Root Mean Squared Error: 0.5103387141563154 
    
    None
    

R2 score based on different types of data tranformations

Just scaling the values to be between 0, 1

Number of PC chosen | LinearRegression | RandomForest 
------------------- | ---------------- | ------------ 
Original data       | 0.5351261336554  | 0.3348235144
Standardize features by removing the mean and scaling to unit variance

Number of PC chosen | LinearRegression | RandomForest 
------------------- | ---------------- | ------------ 
Original data       | 0.5943232652466  | 0.7978303716
Boxcox tranformation of features

Number of PC chosen | LinearRegression | RandomForest 
------------------- | ---------------- | ------------ 
Original data       | 0.642378145744  | 0.83112448250

# Conclusions
The random forest model trained on the boxcox tranformed signals gave the best model with an R2 score of 83. The models trained on the PCA tranformed signals gave a poor result even if the chosen two accounted for ~96% of the data variance.

