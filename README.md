
<html>


<font size = 8>
<font color = 'green'>
<p><center> Sentimental Analysis</center> </p> 
    </font>
</font>
    
   <br><br>
<font size = 3>
        
<p >This is a Data Science project for Ubiqum code academy about sentimental analysis towards and iphone or samsung galaxy using AWS as mean to gather the data.<br>
    <br>
&nbsp;&nbsp;&nbsp;&nbsp;Instead of doing a normal word report I tought it would be more fun doing this in a jupyer notebook, so it can be used for reference for future students wanting to do this project in python.
&nbsp;I will be explaining every step of how to set-up stuff in python liek for example paralell processing, and also referencing the resources I used to perform this task     

</font>
</p>



<br><br>
<br>

<div class="alert alert-warning">
<b>NOTE</b> This are my   <a href="https://github.com/ZordoC">Github</a>
              <a href="https://www.linkedin.com/in/jose-conceicao-050002173/">Linkdin</a> 
  links, where all my projects and contact information are    
</div>


```python
# The ones that are always present with us 

import pandas as pd 
import numpy as np 
import time 

# Paralel processing 
import os 
import multiprocessing as mp
from multiprocessing import Pool

#pre processing 


# modelling 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree




# Visualization 

import matplotlib.pyplot as plt
from IPython.display import display

# Markdown editing 
from IPython.display import HTML
```


```python
from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
Show and Hide Code , click <a href="javascript:code_toggle()">here</a>.''')


```




<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
Show and Hide Code , click <a href="javascript:code_toggle()">here</a>.



<html> 


<font size = 6>
<font color = 'green'>
<p> <center>Setting up Paralell processing</center> </p> 
    </font>
</font>
    
  
 
    
  <font size =  3 >
   
  <br>
Setting up paralel processing is a little more tricky than in R,but it's still pretty simple I followed this  <a href="https://www.youtube.com/watch?v=u2jTn-Gj2Xw&t=706s">youtube video  </a> <br>
&nbsp; I just used the pool function and it worked perfeclty fine. <br>
&nbsp; Just don't use this in every cell code because for small computational tasks is faster to use serial processing because of the time it takes to active all cores somtimes is bigger than the task itself, all details are explained in the video. <br>
&nbsp; Use mp.cpu_count() to know how many processors your machine has 



```python
print("Number of processors: ", mp.cpu_count())
```

    Number of processors:  4



```python
def sum_square(number):
    s=0
    for i  in range(number):
        s+= i*i
    return s 




def sum_square_with_mp(numbers):

    start_time = time.time()
    p = Pool(4)
    result = p.map(sum_square,numbers)

    p.close() 
    p.join()

    end_time = time.time() -start_time
    print(f"Processing {len(numbers)} numbers took {end_time} time using multiprocessing.")


def sum_square_no_mp(numbers):
    start_time = time.time()
    result = []
    for i in numbers:
        result.append(sum_square(i))
    end_time = time.time() - start_time
    
    print(f"Processing {len(numbers)} numbers took {end_time} time using serial processing.")



```


```python
numbers = range(10000)
sum_square_with_mp(numbers)
sum_square_no_mp(numbers)
```

    Processing 10000 numbers took 2.6664981842041016 time using multiprocessing.
    Processing 10000 numbers took 4.710014343261719 time using serial processing.



```python
numbers = range(1000)
sum_square_with_mp(numbers)
sum_square_no_mp(numbers)
```

    Processing 1000 numbers took 0.14234185218811035 time using multiprocessing.
    Processing 1000 numbers took 0.04566764831542969 time using serial processing.


<html> 
       <font size = 5, color = green >
            
<h1><center> Pre-processing </center></h1>
        
   </font>
 <font size = 3 >
    
 <p> &nbsp;&nbsp; &nbsp;&nbsp; First things first , we need to upload our data-sets : iphone sentiments and galaxy sentiments

   
   


```python
samsung_df = pd.read_csv(r"galaxy_smallmatrix_labeled_9d.csv")
iphone_df = pd.read_csv(r"iphone_smallmatrix_labeled_9d.csv")


iphone_df.head()

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
      <th>iphone</th>
      <th>samsunggalaxy</th>
      <th>sonyxperia</th>
      <th>nokialumina</th>
      <th>htcphone</th>
      <th>ios</th>
      <th>googleandroid</th>
      <th>iphonecampos</th>
      <th>samsungcampos</th>
      <th>sonycampos</th>
      <th>...</th>
      <th>sonyperunc</th>
      <th>nokiaperunc</th>
      <th>htcperunc</th>
      <th>iosperpos</th>
      <th>googleperpos</th>
      <th>iosperneg</th>
      <th>googleperneg</th>
      <th>iosperunc</th>
      <th>googleperunc</th>
      <th>iphonesentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>1</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>2</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>3</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>4</th>
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
      <td>...</td>
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
    </tr>
  </tbody>
</table>
<p>5 rows × 59 columns</p>
</div>




```python
samsung_df.head()
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
      <th>iphone</th>
      <th>samsunggalaxy</th>
      <th>sonyxperia</th>
      <th>nokialumina</th>
      <th>htcphone</th>
      <th>ios</th>
      <th>googleandroid</th>
      <th>iphonecampos</th>
      <th>samsungcampos</th>
      <th>sonycampos</th>
      <th>...</th>
      <th>sonyperunc</th>
      <th>nokiaperunc</th>
      <th>htcperunc</th>
      <th>iosperpos</th>
      <th>googleperpos</th>
      <th>iosperneg</th>
      <th>googleperneg</th>
      <th>iosperunc</th>
      <th>googleperunc</th>
      <th>galaxysentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
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
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
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
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>4</th>
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
      <td>...</td>
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
    </tr>
  </tbody>
</table>
<p>5 rows × 59 columns</p>
</div>




 <font size = 6, color = green >
<center> Missing Data </center>

<font size = 3 >
    
 <p> &nbsp;&nbsp; &nbsp;&nbsp; I have some old functions that I used before, I think this will work for any data set, I like to have functions it just makes the code more easy to read, next step is built my own library with all this functions, and then call them like we do with normal libraries, "import sklearn" , or "rom sklearn.metrics import confusion_matrix"

   


```python
def missing_percentage_data(df):
    
    
    missing_values_count = df.isnull().sum()

    total_cells = np.product(df.shape)
    
    total_missing = missing_values_count.sum()

    missing_percent = (total_missing/total_cells) * 100

    print('Percent of missing data of = {}%'.format(missing_percent))
    
    return 
```


```python
missing_percentage_data(samsung_df)
missing_percentage_data(iphone_df)
```

    Percent of missing data of = 0.0%
    Percent of missing data of = 0.0%



 <font size = 6, color = green >
<center> Correlation Matrix </center>

<font size = 3 >
    
 <p> &nbsp;&nbsp; &nbsp;&nbsp; I always like checking correlations and maybe remove some attributes that may cause bias in the future, I also have a function for that , I'll be honest I found this one on stack overflow, I tried to built it myself, but sometimes we have to prioritize our work and learning, and spending one afternoon trying to build a function is not productive to our goal as Data Analysts 

   


```python
def correlation(df, threshold):
    df = df.copy(deep=True) # This is very important, if you dont do this you will update the df you passed through the function
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in df.columns:
                    del df[colname] # deleting the column from the dataset

    return df
```


```python
plt.matshow(iphone_df.corr())
plt.show()
```


![png](output_18_0.png)



```python
iphone_df.corr()
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
      <th>iphone</th>
      <th>samsunggalaxy</th>
      <th>sonyxperia</th>
      <th>nokialumina</th>
      <th>htcphone</th>
      <th>ios</th>
      <th>googleandroid</th>
      <th>iphonecampos</th>
      <th>samsungcampos</th>
      <th>sonycampos</th>
      <th>...</th>
      <th>sonyperunc</th>
      <th>nokiaperunc</th>
      <th>htcperunc</th>
      <th>iosperpos</th>
      <th>googleperpos</th>
      <th>iosperneg</th>
      <th>googleperneg</th>
      <th>iosperunc</th>
      <th>googleperunc</th>
      <th>iphonesentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>iphone</th>
      <td>1.000000</td>
      <td>0.019786</td>
      <td>-0.011618</td>
      <td>-0.013423</td>
      <td>-0.002731</td>
      <td>0.922060</td>
      <td>0.107530</td>
      <td>0.078157</td>
      <td>0.057395</td>
      <td>-0.004594</td>
      <td>...</td>
      <td>-0.003045</td>
      <td>-0.009704</td>
      <td>0.011414</td>
      <td>-0.020059</td>
      <td>0.118008</td>
      <td>-0.019081</td>
      <td>0.138742</td>
      <td>-0.020368</td>
      <td>0.067859</td>
      <td>0.014859</td>
    </tr>
    <tr>
      <th>samsunggalaxy</th>
      <td>0.019786</td>
      <td>1.000000</td>
      <td>0.366671</td>
      <td>-0.006088</td>
      <td>0.017899</td>
      <td>-0.044678</td>
      <td>0.236162</td>
      <td>0.030556</td>
      <td>0.252121</td>
      <td>0.145969</td>
      <td>...</td>
      <td>0.037482</td>
      <td>0.007305</td>
      <td>0.044928</td>
      <td>-0.005802</td>
      <td>0.246046</td>
      <td>-0.007839</td>
      <td>0.290975</td>
      <td>-0.015329</td>
      <td>0.142252</td>
      <td>-0.359173</td>
    </tr>
    <tr>
      <th>sonyxperia</th>
      <td>-0.011618</td>
      <td>0.366671</td>
      <td>1.000000</td>
      <td>-0.006350</td>
      <td>0.023682</td>
      <td>-0.023884</td>
      <td>-0.018288</td>
      <td>0.005068</td>
      <td>0.050140</td>
      <td>0.396751</td>
      <td>...</td>
      <td>0.151675</td>
      <td>-0.004253</td>
      <td>-0.004888</td>
      <td>-0.011009</td>
      <td>-0.008467</td>
      <td>-0.010323</td>
      <td>-0.008570</td>
      <td>-0.014802</td>
      <td>-0.007916</td>
      <td>-0.233170</td>
    </tr>
    <tr>
      <th>nokialumina</th>
      <td>-0.013423</td>
      <td>-0.006088</td>
      <td>-0.006350</td>
      <td>1.000000</td>
      <td>0.000673</td>
      <td>-0.002819</td>
      <td>-0.001115</td>
      <td>0.029824</td>
      <td>0.009299</td>
      <td>-0.002754</td>
      <td>...</td>
      <td>-0.001204</td>
      <td>0.648441</td>
      <td>0.023757</td>
      <td>0.030719</td>
      <td>0.006515</td>
      <td>0.032721</td>
      <td>0.000653</td>
      <td>0.052887</td>
      <td>0.007999</td>
      <td>-0.055962</td>
    </tr>
    <tr>
      <th>htcphone</th>
      <td>-0.002731</td>
      <td>0.017899</td>
      <td>0.023682</td>
      <td>0.000673</td>
      <td>1.000000</td>
      <td>-0.005002</td>
      <td>0.016498</td>
      <td>0.006952</td>
      <td>0.010865</td>
      <td>0.010432</td>
      <td>...</td>
      <td>0.005018</td>
      <td>0.000112</td>
      <td>0.021448</td>
      <td>-0.002927</td>
      <td>0.019186</td>
      <td>-0.002758</td>
      <td>0.020726</td>
      <td>-0.002666</td>
      <td>0.013305</td>
      <td>-0.051285</td>
    </tr>
    <tr>
      <th>ios</th>
      <td>0.922060</td>
      <td>-0.044678</td>
      <td>-0.023884</td>
      <td>-0.002819</td>
      <td>-0.005002</td>
      <td>1.000000</td>
      <td>-0.026404</td>
      <td>0.042128</td>
      <td>-0.010741</td>
      <td>-0.009369</td>
      <td>...</td>
      <td>-0.004832</td>
      <td>0.005030</td>
      <td>-0.011930</td>
      <td>0.118278</td>
      <td>-0.016402</td>
      <td>0.112330</td>
      <td>-0.018028</td>
      <td>0.117035</td>
      <td>-0.010233</td>
      <td>0.001656</td>
    </tr>
    <tr>
      <th>googleandroid</th>
      <td>0.107530</td>
      <td>0.236162</td>
      <td>-0.018288</td>
      <td>-0.001115</td>
      <td>0.016498</td>
      <td>-0.026404</td>
      <td>1.000000</td>
      <td>0.104420</td>
      <td>0.315487</td>
      <td>-0.000206</td>
      <td>...</td>
      <td>-0.004135</td>
      <td>-0.001407</td>
      <td>0.109685</td>
      <td>-0.016702</td>
      <td>0.638581</td>
      <td>-0.015825</td>
      <td>0.716515</td>
      <td>-0.016377</td>
      <td>0.371998</td>
      <td>-0.189142</td>
    </tr>
    <tr>
      <th>iphonecampos</th>
      <td>0.078157</td>
      <td>0.030556</td>
      <td>0.005068</td>
      <td>0.029824</td>
      <td>0.006952</td>
      <td>0.042128</td>
      <td>0.104420</td>
      <td>1.000000</td>
      <td>0.062438</td>
      <td>0.045009</td>
      <td>...</td>
      <td>0.019987</td>
      <td>0.014827</td>
      <td>0.067283</td>
      <td>-0.003991</td>
      <td>0.117902</td>
      <td>-0.007060</td>
      <td>0.124355</td>
      <td>-0.001037</td>
      <td>0.073004</td>
      <td>-0.029731</td>
    </tr>
    <tr>
      <th>samsungcampos</th>
      <td>0.057395</td>
      <td>0.252121</td>
      <td>0.050140</td>
      <td>0.009299</td>
      <td>0.010865</td>
      <td>-0.010741</td>
      <td>0.315487</td>
      <td>0.062438</td>
      <td>1.000000</td>
      <td>0.145429</td>
      <td>...</td>
      <td>0.057860</td>
      <td>0.033197</td>
      <td>0.061304</td>
      <td>0.102471</td>
      <td>0.298281</td>
      <td>0.075695</td>
      <td>0.357362</td>
      <td>0.044890</td>
      <td>0.159171</td>
      <td>-0.112743</td>
    </tr>
    <tr>
      <th>sonycampos</th>
      <td>-0.004594</td>
      <td>0.145969</td>
      <td>0.396751</td>
      <td>-0.002754</td>
      <td>0.010432</td>
      <td>-0.009369</td>
      <td>-0.000206</td>
      <td>0.045009</td>
      <td>0.145429</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.378812</td>
      <td>-0.001845</td>
      <td>0.015781</td>
      <td>-0.003118</td>
      <td>0.006673</td>
      <td>-0.002863</td>
      <td>0.008455</td>
      <td>-0.006421</td>
      <td>-0.003434</td>
      <td>-0.090665</td>
    </tr>
    <tr>
      <th>nokiacampos</th>
      <td>-0.008439</td>
      <td>-0.000400</td>
      <td>-0.004232</td>
      <td>0.700415</td>
      <td>0.000465</td>
      <td>0.005425</td>
      <td>0.003284</td>
      <td>0.030817</td>
      <td>0.014860</td>
      <td>-0.001836</td>
      <td>...</td>
      <td>-0.000802</td>
      <td>0.858295</td>
      <td>0.017261</td>
      <td>0.103123</td>
      <td>0.011564</td>
      <td>0.103540</td>
      <td>0.003941</td>
      <td>0.165188</td>
      <td>0.012518</td>
      <td>-0.033375</td>
    </tr>
    <tr>
      <th>htccampos</th>
      <td>0.022717</td>
      <td>0.065274</td>
      <td>0.016507</td>
      <td>0.021295</td>
      <td>0.023189</td>
      <td>-0.012390</td>
      <td>0.148095</td>
      <td>0.623912</td>
      <td>0.090099</td>
      <td>0.058852</td>
      <td>...</td>
      <td>0.018081</td>
      <td>0.010478</td>
      <td>0.253678</td>
      <td>-0.006121</td>
      <td>0.163145</td>
      <td>-0.005761</td>
      <td>0.177230</td>
      <td>-0.006079</td>
      <td>0.100031</td>
      <td>-0.120434</td>
    </tr>
    <tr>
      <th>iphonecamneg</th>
      <td>0.490524</td>
      <td>0.126063</td>
      <td>-0.006715</td>
      <td>0.063245</td>
      <td>0.014155</td>
      <td>0.386966</td>
      <td>0.391802</td>
      <td>0.541340</td>
      <td>0.206020</td>
      <td>0.013254</td>
      <td>...</td>
      <td>0.032570</td>
      <td>0.026550</td>
      <td>0.114716</td>
      <td>-0.012229</td>
      <td>0.417185</td>
      <td>-0.013642</td>
      <td>0.468075</td>
      <td>-0.010749</td>
      <td>0.241003</td>
      <td>-0.083963</td>
    </tr>
    <tr>
      <th>samsungcamneg</th>
      <td>0.142553</td>
      <td>0.342919</td>
      <td>-0.004308</td>
      <td>0.009546</td>
      <td>0.020021</td>
      <td>-0.015273</td>
      <td>0.711403</td>
      <td>0.117451</td>
      <td>0.608840</td>
      <td>0.032897</td>
      <td>...</td>
      <td>0.060837</td>
      <td>0.036543</td>
      <td>0.122048</td>
      <td>0.110073</td>
      <td>0.658644</td>
      <td>0.081294</td>
      <td>0.794282</td>
      <td>0.047045</td>
      <td>0.342120</td>
      <td>-0.185989</td>
    </tr>
    <tr>
      <th>sonycamneg</th>
      <td>-0.001830</td>
      <td>0.031821</td>
      <td>0.345296</td>
      <td>-0.001229</td>
      <td>0.004909</td>
      <td>-0.003854</td>
      <td>0.013539</td>
      <td>0.019994</td>
      <td>0.053985</td>
      <td>0.408991</td>
      <td>...</td>
      <td>0.604012</td>
      <td>-0.000823</td>
      <td>0.026290</td>
      <td>-0.001276</td>
      <td>0.020904</td>
      <td>-0.001166</td>
      <td>0.025126</td>
      <td>-0.002865</td>
      <td>-0.001532</td>
      <td>-0.024826</td>
    </tr>
    <tr>
      <th>nokiacamneg</th>
      <td>-0.009186</td>
      <td>-0.000979</td>
      <td>-0.004467</td>
      <td>0.729434</td>
      <td>0.000191</td>
      <td>0.004651</td>
      <td>-0.001824</td>
      <td>0.026855</td>
      <td>0.014368</td>
      <td>-0.001938</td>
      <td>...</td>
      <td>-0.000847</td>
      <td>0.788927</td>
      <td>0.016234</td>
      <td>0.089002</td>
      <td>0.002719</td>
      <td>0.090311</td>
      <td>-0.000445</td>
      <td>0.143676</td>
      <td>0.003772</td>
      <td>-0.033069</td>
    </tr>
    <tr>
      <th>htccamneg</th>
      <td>0.104613</td>
      <td>0.222777</td>
      <td>-0.012284</td>
      <td>0.037256</td>
      <td>0.036765</td>
      <td>-0.023049</td>
      <td>0.562703</td>
      <td>0.206585</td>
      <td>0.295428</td>
      <td>0.013568</td>
      <td>...</td>
      <td>0.029574</td>
      <td>0.019518</td>
      <td>0.425361</td>
      <td>-0.010934</td>
      <td>0.578325</td>
      <td>-0.009878</td>
      <td>0.652644</td>
      <td>-0.010191</td>
      <td>0.333727</td>
      <td>-0.222972</td>
    </tr>
    <tr>
      <th>iphonecamunc</th>
      <td>0.750403</td>
      <td>-0.010155</td>
      <td>-0.007638</td>
      <td>0.016237</td>
      <td>0.001174</td>
      <td>0.732612</td>
      <td>0.042955</td>
      <td>0.473266</td>
      <td>0.028875</td>
      <td>0.016442</td>
      <td>...</td>
      <td>0.025256</td>
      <td>0.009049</td>
      <td>0.057397</td>
      <td>-0.004920</td>
      <td>0.076916</td>
      <td>-0.008706</td>
      <td>0.074858</td>
      <td>-0.001336</td>
      <td>0.058139</td>
      <td>0.001443</td>
    </tr>
    <tr>
      <th>samsungcamunc</th>
      <td>0.073451</td>
      <td>0.316134</td>
      <td>0.058777</td>
      <td>0.040922</td>
      <td>0.015644</td>
      <td>-0.012390</td>
      <td>0.391433</td>
      <td>0.076943</td>
      <td>0.814799</td>
      <td>0.164043</td>
      <td>...</td>
      <td>0.152542</td>
      <td>0.123181</td>
      <td>0.124516</td>
      <td>0.129012</td>
      <td>0.417375</td>
      <td>0.097355</td>
      <td>0.476690</td>
      <td>0.057612</td>
      <td>0.269432</td>
      <td>-0.138046</td>
    </tr>
    <tr>
      <th>sonycamunc</th>
      <td>-0.003064</td>
      <td>0.104123</td>
      <td>0.376633</td>
      <td>-0.001914</td>
      <td>0.009843</td>
      <td>-0.006484</td>
      <td>-0.006578</td>
      <td>0.029397</td>
      <td>0.098836</td>
      <td>0.528452</td>
      <td>...</td>
      <td>0.567358</td>
      <td>-0.001282</td>
      <td>0.031963</td>
      <td>-0.000890</td>
      <td>-0.003825</td>
      <td>-0.000746</td>
      <td>-0.004204</td>
      <td>-0.004463</td>
      <td>-0.002386</td>
      <td>-0.050327</td>
    </tr>
    <tr>
      <th>nokiacamunc</th>
      <td>-0.008602</td>
      <td>0.005691</td>
      <td>-0.003972</td>
      <td>0.634171</td>
      <td>0.000364</td>
      <td>0.006341</td>
      <td>0.000325</td>
      <td>0.021277</td>
      <td>0.028323</td>
      <td>-0.001723</td>
      <td>...</td>
      <td>-0.000753</td>
      <td>0.958152</td>
      <td>0.015949</td>
      <td>0.108424</td>
      <td>0.005909</td>
      <td>0.108898</td>
      <td>0.001299</td>
      <td>0.173504</td>
      <td>0.006829</td>
      <td>-0.031550</td>
    </tr>
    <tr>
      <th>htccamunc</th>
      <td>0.026138</td>
      <td>0.072964</td>
      <td>0.014249</td>
      <td>0.036124</td>
      <td>0.029152</td>
      <td>-0.015785</td>
      <td>0.166182</td>
      <td>0.321523</td>
      <td>0.104495</td>
      <td>0.056574</td>
      <td>...</td>
      <td>0.050625</td>
      <td>0.018802</td>
      <td>0.601513</td>
      <td>-0.007866</td>
      <td>0.223305</td>
      <td>-0.007102</td>
      <td>0.227577</td>
      <td>-0.005186</td>
      <td>0.162431</td>
      <td>-0.148881</td>
    </tr>
    <tr>
      <th>iphonedispos</th>
      <td>0.052625</td>
      <td>-0.006526</td>
      <td>-0.018121</td>
      <td>0.028316</td>
      <td>0.000253</td>
      <td>0.014377</td>
      <td>0.066953</td>
      <td>0.272587</td>
      <td>0.039427</td>
      <td>0.019617</td>
      <td>...</td>
      <td>0.027681</td>
      <td>0.012382</td>
      <td>0.091895</td>
      <td>0.020232</td>
      <td>0.165576</td>
      <td>0.015293</td>
      <td>0.147023</td>
      <td>0.024767</td>
      <td>0.179686</td>
      <td>0.014547</td>
    </tr>
    <tr>
      <th>samsungdispos</th>
      <td>0.061074</td>
      <td>0.281379</td>
      <td>0.040063</td>
      <td>0.041456</td>
      <td>0.013145</td>
      <td>-0.009906</td>
      <td>0.316132</td>
      <td>0.060476</td>
      <td>0.643692</td>
      <td>0.111287</td>
      <td>...</td>
      <td>0.111113</td>
      <td>0.123591</td>
      <td>0.285206</td>
      <td>0.118107</td>
      <td>0.606458</td>
      <td>0.092014</td>
      <td>0.579951</td>
      <td>0.057288</td>
      <td>0.636103</td>
      <td>-0.099262</td>
    </tr>
    <tr>
      <th>sonydispos</th>
      <td>-0.003827</td>
      <td>0.061360</td>
      <td>0.252589</td>
      <td>-0.001528</td>
      <td>0.006959</td>
      <td>0.004207</td>
      <td>-0.001669</td>
      <td>0.017749</td>
      <td>0.058122</td>
      <td>0.404993</td>
      <td>...</td>
      <td>0.340766</td>
      <td>-0.001024</td>
      <td>0.019407</td>
      <td>0.025392</td>
      <td>0.000158</td>
      <td>0.024833</td>
      <td>0.001709</td>
      <td>-0.003562</td>
      <td>-0.001905</td>
      <td>-0.038635</td>
    </tr>
    <tr>
      <th>nokiadispos</th>
      <td>-0.008202</td>
      <td>0.010248</td>
      <td>-0.003772</td>
      <td>0.650253</td>
      <td>0.000311</td>
      <td>0.003065</td>
      <td>-0.004174</td>
      <td>0.026317</td>
      <td>0.038371</td>
      <td>-0.001636</td>
      <td>...</td>
      <td>-0.000715</td>
      <td>0.836700</td>
      <td>0.014490</td>
      <td>0.079541</td>
      <td>-0.002427</td>
      <td>0.079405</td>
      <td>-0.002668</td>
      <td>0.127264</td>
      <td>-0.001514</td>
      <td>-0.025922</td>
    </tr>
    <tr>
      <th>htcdispos</th>
      <td>0.007125</td>
      <td>0.024839</td>
      <td>0.003299</td>
      <td>0.010554</td>
      <td>0.977538</td>
      <td>-0.005749</td>
      <td>0.057552</td>
      <td>0.067429</td>
      <td>0.032923</td>
      <td>0.016457</td>
      <td>...</td>
      <td>0.015451</td>
      <td>0.005317</td>
      <td>0.135495</td>
      <td>-0.001147</td>
      <td>0.118340</td>
      <td>-0.001013</td>
      <td>0.109239</td>
      <td>0.000579</td>
      <td>0.124018</td>
      <td>-0.060406</td>
    </tr>
    <tr>
      <th>iphonedisneg</th>
      <td>0.175573</td>
      <td>0.017824</td>
      <td>-0.013590</td>
      <td>0.023742</td>
      <td>0.002796</td>
      <td>0.113784</td>
      <td>0.121821</td>
      <td>0.148651</td>
      <td>0.065279</td>
      <td>0.006717</td>
      <td>...</td>
      <td>0.023878</td>
      <td>0.009723</td>
      <td>0.096514</td>
      <td>0.015557</td>
      <td>0.218541</td>
      <td>0.016863</td>
      <td>0.213640</td>
      <td>0.018222</td>
      <td>0.204416</td>
      <td>0.003145</td>
    </tr>
    <tr>
      <th>samsungdisneg</th>
      <td>0.111821</td>
      <td>0.304385</td>
      <td>0.007706</td>
      <td>0.022910</td>
      <td>0.020154</td>
      <td>-0.011706</td>
      <td>0.542440</td>
      <td>0.089813</td>
      <td>0.487871</td>
      <td>0.058932</td>
      <td>...</td>
      <td>0.108932</td>
      <td>0.072670</td>
      <td>0.317065</td>
      <td>0.104504</td>
      <td>0.808753</td>
      <td>0.081942</td>
      <td>0.826604</td>
      <td>0.050730</td>
      <td>0.735526</td>
      <td>-0.139965</td>
    </tr>
    <tr>
      <th>sonydisneg</th>
      <td>-0.002777</td>
      <td>0.006786</td>
      <td>0.163285</td>
      <td>-0.000644</td>
      <td>0.003056</td>
      <td>0.008063</td>
      <td>0.000215</td>
      <td>0.002884</td>
      <td>0.011236</td>
      <td>0.131892</td>
      <td>...</td>
      <td>0.112633</td>
      <td>-0.000431</td>
      <td>0.006683</td>
      <td>0.030273</td>
      <td>0.000580</td>
      <td>0.029531</td>
      <td>0.001907</td>
      <td>-0.001501</td>
      <td>-0.000803</td>
      <td>-0.019956</td>
    </tr>
    <tr>
      <th>nokiadisneg</th>
      <td>-0.008790</td>
      <td>0.005640</td>
      <td>-0.004040</td>
      <td>0.692268</td>
      <td>0.000389</td>
      <td>0.003432</td>
      <td>-0.004470</td>
      <td>0.026307</td>
      <td>0.028454</td>
      <td>-0.001752</td>
      <td>...</td>
      <td>-0.000766</td>
      <td>0.861817</td>
      <td>0.017639</td>
      <td>0.087180</td>
      <td>-0.002599</td>
      <td>0.086986</td>
      <td>-0.002857</td>
      <td>0.139410</td>
      <td>-0.001622</td>
      <td>-0.028759</td>
    </tr>
    <tr>
      <th>htcdisneg</th>
      <td>0.085273</td>
      <td>0.188821</td>
      <td>-0.002138</td>
      <td>0.044222</td>
      <td>0.037653</td>
      <td>-0.019709</td>
      <td>0.447013</td>
      <td>0.110102</td>
      <td>0.238425</td>
      <td>0.037624</td>
      <td>...</td>
      <td>0.068252</td>
      <td>0.023569</td>
      <td>0.549764</td>
      <td>0.000258</td>
      <td>0.704117</td>
      <td>0.001073</td>
      <td>0.698363</td>
      <td>0.009817</td>
      <td>0.643174</td>
      <td>-0.192727</td>
    </tr>
    <tr>
      <th>iphonedisunc</th>
      <td>0.250930</td>
      <td>-0.027879</td>
      <td>-0.017981</td>
      <td>0.002681</td>
      <td>-0.002108</td>
      <td>0.218835</td>
      <td>0.017791</td>
      <td>0.188310</td>
      <td>0.012313</td>
      <td>0.007384</td>
      <td>...</td>
      <td>0.022878</td>
      <td>0.001485</td>
      <td>0.092743</td>
      <td>0.024055</td>
      <td>0.132862</td>
      <td>0.023660</td>
      <td>0.106084</td>
      <td>0.030107</td>
      <td>0.172276</td>
      <td>0.027173</td>
    </tr>
    <tr>
      <th>samsungdisunc</th>
      <td>0.038727</td>
      <td>0.190038</td>
      <td>0.026314</td>
      <td>0.046896</td>
      <td>0.009413</td>
      <td>-0.005951</td>
      <td>0.188924</td>
      <td>0.035791</td>
      <td>0.389689</td>
      <td>0.089489</td>
      <td>...</td>
      <td>0.129397</td>
      <td>0.136216</td>
      <td>0.344839</td>
      <td>0.078623</td>
      <td>0.594372</td>
      <td>0.063341</td>
      <td>0.512732</td>
      <td>0.039951</td>
      <td>0.738457</td>
      <td>-0.059548</td>
    </tr>
    <tr>
      <th>sonydisunc</th>
      <td>-0.004553</td>
      <td>0.060556</td>
      <td>0.295428</td>
      <td>-0.001383</td>
      <td>0.005268</td>
      <td>0.000626</td>
      <td>-0.004753</td>
      <td>0.019403</td>
      <td>0.067668</td>
      <td>0.388804</td>
      <td>...</td>
      <td>0.476597</td>
      <td>-0.000927</td>
      <td>0.037009</td>
      <td>0.014600</td>
      <td>-0.002764</td>
      <td>0.014310</td>
      <td>-0.003038</td>
      <td>-0.003225</td>
      <td>-0.001724</td>
      <td>-0.032137</td>
    </tr>
    <tr>
      <th>nokiadisunc</th>
      <td>-0.007588</td>
      <td>0.014661</td>
      <td>-0.003233</td>
      <td>0.491332</td>
      <td>-0.000066</td>
      <td>0.006110</td>
      <td>-0.003577</td>
      <td>0.009608</td>
      <td>0.046812</td>
      <td>-0.001402</td>
      <td>...</td>
      <td>-0.000613</td>
      <td>0.923934</td>
      <td>0.007326</td>
      <td>0.102037</td>
      <td>-0.002080</td>
      <td>0.102012</td>
      <td>-0.002286</td>
      <td>0.162690</td>
      <td>-0.001298</td>
      <td>-0.023972</td>
    </tr>
    <tr>
      <th>htcdisunc</th>
      <td>0.024322</td>
      <td>0.071746</td>
      <td>0.010003</td>
      <td>0.021114</td>
      <td>0.029195</td>
      <td>-0.012652</td>
      <td>0.147068</td>
      <td>0.156063</td>
      <td>0.086766</td>
      <td>0.055055</td>
      <td>...</td>
      <td>0.083275</td>
      <td>0.010655</td>
      <td>0.721407</td>
      <td>0.006433</td>
      <td>0.483030</td>
      <td>0.006752</td>
      <td>0.406569</td>
      <td>0.019677</td>
      <td>0.593494</td>
      <td>-0.132953</td>
    </tr>
    <tr>
      <th>iphoneperpos</th>
      <td>-0.009508</td>
      <td>-0.003169</td>
      <td>-0.028717</td>
      <td>0.033345</td>
      <td>0.000121</td>
      <td>-0.021953</td>
      <td>0.106061</td>
      <td>0.348332</td>
      <td>0.056272</td>
      <td>0.009152</td>
      <td>...</td>
      <td>0.036380</td>
      <td>0.015281</td>
      <td>0.123390</td>
      <td>0.210343</td>
      <td>0.240267</td>
      <td>0.224650</td>
      <td>0.218848</td>
      <td>0.211809</td>
      <td>0.237625</td>
      <td>0.029638</td>
    </tr>
    <tr>
      <th>samsungperpos</th>
      <td>0.051538</td>
      <td>0.242866</td>
      <td>0.020914</td>
      <td>0.017459</td>
      <td>0.009711</td>
      <td>-0.002131</td>
      <td>0.270355</td>
      <td>0.045221</td>
      <td>0.793899</td>
      <td>0.046923</td>
      <td>...</td>
      <td>0.057896</td>
      <td>0.055507</td>
      <td>0.189565</td>
      <td>0.274209</td>
      <td>0.444302</td>
      <td>0.214228</td>
      <td>0.441229</td>
      <td>0.137057</td>
      <td>0.427542</td>
      <td>-0.081063</td>
    </tr>
    <tr>
      <th>sonyperpos</th>
      <td>-0.006327</td>
      <td>0.067489</td>
      <td>0.266142</td>
      <td>-0.001919</td>
      <td>0.004812</td>
      <td>-0.004091</td>
      <td>0.000836</td>
      <td>0.013944</td>
      <td>0.047395</td>
      <td>0.387311</td>
      <td>...</td>
      <td>0.735802</td>
      <td>-0.001285</td>
      <td>0.014073</td>
      <td>0.005758</td>
      <td>0.007712</td>
      <td>0.005731</td>
      <td>0.008070</td>
      <td>-0.004473</td>
      <td>-0.002392</td>
      <td>-0.038913</td>
    </tr>
    <tr>
      <th>nokiaperpos</th>
      <td>-0.010509</td>
      <td>0.001846</td>
      <td>-0.004606</td>
      <td>0.737457</td>
      <td>0.000454</td>
      <td>0.002261</td>
      <td>-0.002300</td>
      <td>0.021178</td>
      <td>0.021581</td>
      <td>-0.001998</td>
      <td>...</td>
      <td>-0.000873</td>
      <td>0.917333</td>
      <td>0.016373</td>
      <td>0.084948</td>
      <td>0.002051</td>
      <td>0.084529</td>
      <td>-0.000824</td>
      <td>0.135942</td>
      <td>0.003141</td>
      <td>-0.041595</td>
    </tr>
    <tr>
      <th>htcperpos</th>
      <td>0.030621</td>
      <td>0.088289</td>
      <td>0.004677</td>
      <td>0.039113</td>
      <td>0.030909</td>
      <td>-0.018167</td>
      <td>0.209414</td>
      <td>0.287085</td>
      <td>0.115132</td>
      <td>0.021326</td>
      <td>...</td>
      <td>0.026076</td>
      <td>0.020553</td>
      <td>0.849739</td>
      <td>-0.002803</td>
      <td>0.380278</td>
      <td>-0.002767</td>
      <td>0.358458</td>
      <td>-0.000189</td>
      <td>0.368327</td>
      <td>-0.178427</td>
    </tr>
    <tr>
      <th>iphoneperneg</th>
      <td>0.013863</td>
      <td>0.045963</td>
      <td>-0.028774</td>
      <td>0.033735</td>
      <td>0.004285</td>
      <td>-0.012566</td>
      <td>0.212525</td>
      <td>0.151919</td>
      <td>0.112508</td>
      <td>0.006280</td>
      <td>...</td>
      <td>0.042156</td>
      <td>0.015347</td>
      <td>0.140184</td>
      <td>0.247457</td>
      <td>0.345247</td>
      <td>0.282779</td>
      <td>0.348685</td>
      <td>0.255736</td>
      <td>0.296227</td>
      <td>-0.004804</td>
    </tr>
    <tr>
      <th>samsungperneg</th>
      <td>0.115130</td>
      <td>0.303560</td>
      <td>-0.001931</td>
      <td>0.017354</td>
      <td>0.017457</td>
      <td>-0.007168</td>
      <td>0.558090</td>
      <td>0.092030</td>
      <td>0.546670</td>
      <td>0.034149</td>
      <td>...</td>
      <td>0.060809</td>
      <td>0.057204</td>
      <td>0.271245</td>
      <td>0.202892</td>
      <td>0.758411</td>
      <td>0.161560</td>
      <td>0.796365</td>
      <td>0.103158</td>
      <td>0.641229</td>
      <td>-0.138657</td>
    </tr>
    <tr>
      <th>sonyperneg</th>
      <td>-0.003625</td>
      <td>0.009977</td>
      <td>0.122407</td>
      <td>-0.000948</td>
      <td>0.001113</td>
      <td>-0.002902</td>
      <td>0.005657</td>
      <td>0.007034</td>
      <td>0.019366</td>
      <td>0.182829</td>
      <td>...</td>
      <td>0.668018</td>
      <td>-0.000635</td>
      <td>0.006295</td>
      <td>0.000040</td>
      <td>0.010539</td>
      <td>0.000099</td>
      <td>0.012140</td>
      <td>-0.002210</td>
      <td>-0.001182</td>
      <td>-0.030850</td>
    </tr>
    <tr>
      <th>nokiaperneg</th>
      <td>-0.010781</td>
      <td>0.000481</td>
      <td>-0.004699</td>
      <td>0.736453</td>
      <td>0.000462</td>
      <td>0.002322</td>
      <td>-0.003226</td>
      <td>0.017987</td>
      <td>0.018696</td>
      <td>-0.002038</td>
      <td>...</td>
      <td>-0.000891</td>
      <td>0.905222</td>
      <td>0.017733</td>
      <td>0.079553</td>
      <td>0.000515</td>
      <td>0.079819</td>
      <td>-0.001606</td>
      <td>0.128002</td>
      <td>0.001635</td>
      <td>-0.044219</td>
    </tr>
    <tr>
      <th>htcperneg</th>
      <td>0.075975</td>
      <td>0.178410</td>
      <td>-0.012083</td>
      <td>0.050051</td>
      <td>0.033942</td>
      <td>-0.021576</td>
      <td>0.433411</td>
      <td>0.109392</td>
      <td>0.231172</td>
      <td>0.009013</td>
      <td>...</td>
      <td>0.021705</td>
      <td>0.026814</td>
      <td>0.659652</td>
      <td>-0.003590</td>
      <td>0.628876</td>
      <td>-0.002524</td>
      <td>0.638941</td>
      <td>0.000362</td>
      <td>0.539902</td>
      <td>-0.209196</td>
    </tr>
    <tr>
      <th>iphoneperunc</th>
      <td>-0.016037</td>
      <td>-0.017389</td>
      <td>-0.028220</td>
      <td>0.020197</td>
      <td>0.000194</td>
      <td>-0.015482</td>
      <td>0.056676</td>
      <td>0.187260</td>
      <td>0.031845</td>
      <td>0.008176</td>
      <td>...</td>
      <td>0.050653</td>
      <td>0.012553</td>
      <td>0.171436</td>
      <td>0.166660</td>
      <td>0.242735</td>
      <td>0.179411</td>
      <td>0.196254</td>
      <td>0.181783</td>
      <td>0.297140</td>
      <td>0.037200</td>
    </tr>
    <tr>
      <th>samsungperunc</th>
      <td>0.046822</td>
      <td>0.184775</td>
      <td>0.008008</td>
      <td>0.035274</td>
      <td>0.010644</td>
      <td>-0.004770</td>
      <td>0.221726</td>
      <td>0.040154</td>
      <td>0.487767</td>
      <td>0.053436</td>
      <td>...</td>
      <td>0.091928</td>
      <td>0.103767</td>
      <td>0.346705</td>
      <td>0.102804</td>
      <td>0.616442</td>
      <td>0.083218</td>
      <td>0.541504</td>
      <td>0.053897</td>
      <td>0.739887</td>
      <td>-0.057920</td>
    </tr>
    <tr>
      <th>sonyperunc</th>
      <td>-0.003045</td>
      <td>0.037482</td>
      <td>0.151675</td>
      <td>-0.001204</td>
      <td>0.005018</td>
      <td>-0.004832</td>
      <td>-0.004135</td>
      <td>0.019987</td>
      <td>0.057860</td>
      <td>0.378812</td>
      <td>...</td>
      <td>1.000000</td>
      <td>-0.000806</td>
      <td>0.033233</td>
      <td>-0.002861</td>
      <td>-0.002405</td>
      <td>-0.002711</td>
      <td>-0.002643</td>
      <td>-0.002806</td>
      <td>-0.001500</td>
      <td>-0.018084</td>
    </tr>
    <tr>
      <th>nokiaperunc</th>
      <td>-0.009704</td>
      <td>0.007305</td>
      <td>-0.004253</td>
      <td>0.648441</td>
      <td>0.000112</td>
      <td>0.005030</td>
      <td>-0.001407</td>
      <td>0.014827</td>
      <td>0.033197</td>
      <td>-0.001845</td>
      <td>...</td>
      <td>-0.000806</td>
      <td>1.000000</td>
      <td>0.012363</td>
      <td>0.098336</td>
      <td>0.003180</td>
      <td>0.098859</td>
      <td>-0.000137</td>
      <td>0.157714</td>
      <td>0.004180</td>
      <td>-0.036167</td>
    </tr>
    <tr>
      <th>htcperunc</th>
      <td>0.011414</td>
      <td>0.044928</td>
      <td>-0.004888</td>
      <td>0.023757</td>
      <td>0.021448</td>
      <td>-0.011930</td>
      <td>0.109685</td>
      <td>0.067283</td>
      <td>0.061304</td>
      <td>0.015781</td>
      <td>...</td>
      <td>0.033233</td>
      <td>0.012363</td>
      <td>1.000000</td>
      <td>0.000969</td>
      <td>0.333022</td>
      <td>0.000673</td>
      <td>0.280893</td>
      <td>0.008437</td>
      <td>0.394552</td>
      <td>-0.114171</td>
    </tr>
    <tr>
      <th>iosperpos</th>
      <td>-0.020059</td>
      <td>-0.005802</td>
      <td>-0.011009</td>
      <td>0.030719</td>
      <td>-0.002927</td>
      <td>0.118278</td>
      <td>-0.016702</td>
      <td>-0.003991</td>
      <td>0.102471</td>
      <td>-0.003118</td>
      <td>...</td>
      <td>-0.002861</td>
      <td>0.098336</td>
      <td>0.000969</td>
      <td>1.000000</td>
      <td>-0.009712</td>
      <td>0.932382</td>
      <td>-0.010676</td>
      <td>0.905079</td>
      <td>-0.006060</td>
      <td>-0.015758</td>
    </tr>
    <tr>
      <th>googleperpos</th>
      <td>0.118008</td>
      <td>0.246046</td>
      <td>-0.008467</td>
      <td>0.006515</td>
      <td>0.019186</td>
      <td>-0.016402</td>
      <td>0.638581</td>
      <td>0.117902</td>
      <td>0.298281</td>
      <td>0.006673</td>
      <td>...</td>
      <td>-0.002405</td>
      <td>0.003180</td>
      <td>0.333022</td>
      <td>-0.009712</td>
      <td>1.000000</td>
      <td>-0.009203</td>
      <td>0.957410</td>
      <td>-0.009524</td>
      <td>0.887033</td>
      <td>-0.137261</td>
    </tr>
    <tr>
      <th>iosperneg</th>
      <td>-0.019081</td>
      <td>-0.007839</td>
      <td>-0.010323</td>
      <td>0.032721</td>
      <td>-0.002758</td>
      <td>0.112330</td>
      <td>-0.015825</td>
      <td>-0.007060</td>
      <td>0.075695</td>
      <td>-0.002863</td>
      <td>...</td>
      <td>-0.002711</td>
      <td>0.098859</td>
      <td>0.000673</td>
      <td>0.932382</td>
      <td>-0.009203</td>
      <td>1.000000</td>
      <td>-0.010115</td>
      <td>0.899819</td>
      <td>-0.005742</td>
      <td>-0.010179</td>
    </tr>
    <tr>
      <th>googleperneg</th>
      <td>0.138742</td>
      <td>0.290975</td>
      <td>-0.008570</td>
      <td>0.000653</td>
      <td>0.020726</td>
      <td>-0.018028</td>
      <td>0.716515</td>
      <td>0.124355</td>
      <td>0.357362</td>
      <td>0.008455</td>
      <td>...</td>
      <td>-0.002643</td>
      <td>-0.000137</td>
      <td>0.280893</td>
      <td>-0.010676</td>
      <td>0.957410</td>
      <td>-0.010115</td>
      <td>1.000000</td>
      <td>-0.010468</td>
      <td>0.756118</td>
      <td>-0.163919</td>
    </tr>
    <tr>
      <th>iosperunc</th>
      <td>-0.020368</td>
      <td>-0.015329</td>
      <td>-0.014802</td>
      <td>0.052887</td>
      <td>-0.002666</td>
      <td>0.117035</td>
      <td>-0.016377</td>
      <td>-0.001037</td>
      <td>0.044890</td>
      <td>-0.006421</td>
      <td>...</td>
      <td>-0.002806</td>
      <td>0.157714</td>
      <td>0.008437</td>
      <td>0.905079</td>
      <td>-0.009524</td>
      <td>0.899819</td>
      <td>-0.010468</td>
      <td>1.000000</td>
      <td>-0.005942</td>
      <td>-0.011787</td>
    </tr>
    <tr>
      <th>googleperunc</th>
      <td>0.067859</td>
      <td>0.142252</td>
      <td>-0.007916</td>
      <td>0.007999</td>
      <td>0.013305</td>
      <td>-0.010233</td>
      <td>0.371998</td>
      <td>0.073004</td>
      <td>0.159171</td>
      <td>-0.003434</td>
      <td>...</td>
      <td>-0.001500</td>
      <td>0.004180</td>
      <td>0.394552</td>
      <td>-0.006060</td>
      <td>0.887033</td>
      <td>-0.005742</td>
      <td>0.756118</td>
      <td>-0.005942</td>
      <td>1.000000</td>
      <td>-0.070284</td>
    </tr>
    <tr>
      <th>iphonesentiment</th>
      <td>0.014859</td>
      <td>-0.359173</td>
      <td>-0.233170</td>
      <td>-0.055962</td>
      <td>-0.051285</td>
      <td>0.001656</td>
      <td>-0.189142</td>
      <td>-0.029731</td>
      <td>-0.112743</td>
      <td>-0.090665</td>
      <td>...</td>
      <td>-0.018084</td>
      <td>-0.036167</td>
      <td>-0.114171</td>
      <td>-0.015758</td>
      <td>-0.137261</td>
      <td>-0.010179</td>
      <td>-0.163919</td>
      <td>-0.011787</td>
      <td>-0.070284</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>59 rows × 59 columns</p>
</div>




 <font size = 6, color = green >
<center> Near 0 variance </center>

<font size = 3 >
    
 <p>  I created a function that takes out columns with variance less than the treshhold value specified.<br>
&nbsp;&nbsp; &nbsp;This one I did it myself, I'm sure you can find more elegant solutions online, but I like making this simple functions because they I learn a lot by designing them 
   </p>


```python
def near_zero_var(df,tresh):
    df = df.copy(deep=True)
    cols = df.columns
    for i in range(len(cols)):
        if df[cols[i]].var() <= tresh :
            del df[cols[i]] 
    
    return df 
    
```


 <font size = 6, color = green >
<center> Recursive feature elimination using cross validation </center>

<font size = 3 >
    
 <p>&nbsp;&nbsp;&nbsp;  With this method we basically start a model with all variables (n), remove a variable and see models performance metrics, create another model now with the n-1 variables see performance metrics and so on.<br>
 &nbsp;&nbsp;&nbsp; This is basically a brute force method that tests all possible feature combination and ranks the features based on their importance.    
   </p>
   
   <br>
   <br>

<html>

<font face="verdana" color="blue" size = 4 > 

import time <br>
from multiprocessing import Pool <br>
from sklearn.ensemble import RandomForestClassifier <br>
from sklearn.feature_selection import RFECV <br>
from sklearn.svm import SVR <br>


start_time= time.time()
p = Pool(4)


estimator = SVR(kernel="linear")

selector = RFECV(estimator, step=1, cv=5)

selector = selector.fit(X_iphone, Y_iphone)

selector.support_ 

selector.ranking_ 



p.close()

p.join()

end_time =  time.time() - start_time 

print("--- Run Time is : %s mins--- " % np.round(((time.time()-start_time)/60),2))

<br>
<br>


<div class="alert alert-warning">
<b>NOTE</b> This is in markdown syntax because it takes around 160 minutes even with paralell processing     
</div>


<font size = 3 >
    
 <p>&nbsp;&nbsp;&nbsp; Anyways here are my results for both matrices :   </p>
   
   <br>
   <br>


```python
samsungvarimp =([20,  1,  1,  1,  1,  6,  1, 29, 11, 28,  1, 10, 25, 17,  1,  1,  5,
       30,  2, 16,  1,  3, 21,  7,  1,  1,  1, 23,  8,  9, 12,  1, 19,  1,
        1, 13,  1, 22,  1,  1,  1,  1, 26,  1,  1,  1, 18, 24,  1,  1,  1,
        1, 15,  4, 27,  1, 14,  1])



print(samsungvarimp)


```

    [20, 1, 1, 1, 1, 6, 1, 29, 11, 28, 1, 10, 25, 17, 1, 1, 5, 30, 2, 16, 1, 3, 21, 7, 1, 1, 1, 23, 8, 9, 12, 1, 19, 1, 1, 13, 1, 22, 1, 1, 1, 1, 26, 1, 1, 1, 18, 24, 1, 1, 1, 1, 15, 4, 27, 1, 14, 1]



```python
varimpihpone = ([21,  1,  1,  1,  1, 12,  1, 33,  9, 25,  1,  7, 31,  1,  1, 34,  6,
       32, 20, 15,  1, 16, 22, 14,  1, 28,  1, 26,  2, 13,  1,  1, 19,  8,
        1, 10,  1, 23,  3,  1,  1,  1, 30,  1,  1, 11, 27, 24,  1,  1,  1,
        1, 18,  1, 29,  5, 17,  4])


print(varimpihpone)

```

    [21, 1, 1, 1, 1, 12, 1, 33, 9, 25, 1, 7, 31, 1, 1, 34, 6, 32, 20, 15, 1, 16, 22, 14, 1, 28, 1, 26, 2, 13, 1, 1, 19, 8, 1, 10, 1, 23, 3, 1, 1, 1, 30, 1, 1, 11, 27, 24, 1, 1, 1, 1, 18, 1, 29, 5, 17, 4]



```python
def most_imp_vars(df,varimparray):
    df_copy= df.copy()
    for i in varimparray:
        if varimparray[i] != 1:
            print(df_copy.columns[i])
            del df_copy[df_copy.columns[i]]
            
    return df_copy 
    
```

<html> 
       <font size = 5, color = green >
            
<h1><center> Modelling </center></h1>
        
   </font>


```python
def X_Y_split(df,label):
    
    X= df.loc[:,df.columns != label ]
    y = df[label]
    
    return X,y 

    
    
```

<font size = 3 >
    
 <p>
    
&nbsp; Our label is the sentiment column (iphone and galaxy) , which are evaluated from 0 to 5, the exact description of each level  is on the plan of atack, but to sum up 0 is very bad and 5 is excellent.<br> <br>
&nbsp; Above there is a simple function I designed to split label and predictors.For train and test splits we will just use the sklearn function called train_test_split. 
&nbsp;<br><br>
&nbsp; Next I use a range of different models the orignal dataset to see which performs best 

</p>
   
   
   <br>


```python
X_iphone,y_iphone = X_Y_split(iphone_df,'iphonesentiment')



display(X_iphone.head())

display(pd.DataFrame(y_iphone.head()))
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
      <th>iphone</th>
      <th>samsunggalaxy</th>
      <th>sonyxperia</th>
      <th>nokialumina</th>
      <th>htcphone</th>
      <th>ios</th>
      <th>googleandroid</th>
      <th>iphonecampos</th>
      <th>samsungcampos</th>
      <th>sonycampos</th>
      <th>...</th>
      <th>samsungperunc</th>
      <th>sonyperunc</th>
      <th>nokiaperunc</th>
      <th>htcperunc</th>
      <th>iosperpos</th>
      <th>googleperpos</th>
      <th>iosperneg</th>
      <th>googleperneg</th>
      <th>iosperunc</th>
      <th>googleperunc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>1</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>2</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>3</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>4</th>
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
      <td>...</td>
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
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>



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
      <th>iphonesentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
#Create training and testing sets with a 70/30 split using createDataPartition.

X_iphone,y_iphone = X_Y_split(iphone_df,'iphonesentiment')

X_train_iphone , X_test_iphone , y_train_iphone , y_test_iphone =  train_test_split(X_iphone, y_iphone, test_size=0.33, random_state=42)



```


 <font size = 5, color = green >
<center> Random Forrest  </center>
    
  <br>


```python
start_time = time.time()

rf = RandomForestClassifier()

rf.fit(X_train_iphone,y_train_iphone)

predictions = rf.predict(X_test_iphone)

end_time = start_time - time.time()

print("--- Run Time for Random Forrest  is : %s mins--- \n " % np.round(((time.time()-start_time)/60),2))
print("Accuracy : ",accuracy_score(predictions,y_test_iphone))
print("\n")

print("Kappa's : ",cohen_kappa_score(predictions,y_test_iphone))

print("\n")
print("Confusion Matrix : \n \n  ",confusion_matrix(predictions,y_test_iphone) )

```

    --- Run Time for Random Forrest  is : 0.02 mins--- 
     
    Accuracy :  0.7648295189163942
    
    
    Kappa's :  0.5446469878776314
    
    
    Confusion Matrix : 
     
       [[ 387    1    1    1    5    7]
     [   1    0    0    0    0    2]
     [   0    0   21    0    1    6]
     [   2    0    1  272    2   10]
     [   5    1    1    5  155   35]
     [ 220  112  138  121  329 2440]]



 <font size = 5, color = green >
<center> SVM  </center>
    
  <br>


```python


start_time = time.time()

svc = SVC()

svc.fit(X_train_iphone,y_train_iphone)

predictions = svc.predict(X_test_iphone)

end_time = start_time - time.time()

print("--- Run Time for SVM is : %s mins--- \n " % np.round(((time.time()-start_time)/60),2))
print("Accuracy : ",accuracy_score(predictions,y_test_iphone))
print("\n")

print("Kappa's : ",cohen_kappa_score(predictions,y_test_iphone))

print("\n")
print("Confusion Matrix : \n \n  ",confusion_matrix(predictions,y_test_iphone) )

```

    --- Run Time for SVM is : 0.17 mins--- 
     
    Accuracy :  0.7223260158804297
    
    
    Kappa's :  0.44147337960791777
    
    
    Confusion Matrix : 
     
       [[ 392    0    0   19   16   48]
     [   0    0    0    0    0    0]
     [   0    0    0    0    0    0]
     [   2    1   20  119    2    0]
     [   1    0    1    1  135    5]
     [ 220  113  141  260  339 2447]]



 <font size = 5, color = green >
<center> K-NN </center>
    
  <br>


```python
from sklearn.neighbors import KNeighborsClassifier

start_time = time.time()

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train_iphone,y_train_iphone)

predictions = knn.predict(X_test_iphone)

end_time = start_time - time.time()

print("--- Run Time for KNN is : %s mins--- \n " % np.round(((time.time()-start_time)/60),2))
print("Accuracy : ",accuracy_score(predictions,y_test_iphone))
print("\n")

print("Kappa's : ",cohen_kappa_score(predictions,y_test_iphone))

print("\n")
print("Confusion Matrix : \n \n  ",confusion_matrix(predictions,y_test_iphone) )

```

    --- Run Time for KNN is : 0.05 mins--- 
     
    Accuracy :  0.708780943484353
    
    
    Kappa's :  0.47335990046448917
    
    
    Confusion Matrix : 
     
       [[ 381    2    5    7   18   77]
     [   9    2    4    2    9   66]
     [   0    1   22    0    4   18]
     [   3    2    3  273    7   27]
     [  19    6    8    4  153  108]
     [ 203  101  120  113  301 2204]]



 <font size = 5, color = green >
<center> Tree</center>
    
  <br>


```python

start_time = time.time()

t =tree.DecisionTreeClassifier()

t.fit(X_train_iphone,y_train_iphone)

predictions = t.predict(X_test_iphone)

end_time = start_time - time.time()

print("--- Run Time for decison tree is : %s mins--- \n " % np.round(((time.time()-start_time)/60),2))
print("Accuracy : ",accuracy_score(predictions,y_test_iphone))
print("\n")

print("Kappa's : ",cohen_kappa_score(predictions,y_test_iphone))

print("\n")
print("Confusion Matrix : \n \n  ",confusion_matrix(predictions,y_test_iphone) )

```

    --- Run Time for decison tree is : 0.0 mins--- 
     
    Accuracy :  0.7417094815506773
    
    
    Kappa's :  0.5151552255296251
    
    
    Confusion Matrix : 
     
       [[ 382    1    6    2   13   32]
     [   2    0    0    1    2   13]
     [   2    0   20    1    6   28]
     [   4    0    1  273    4   28]
     [   7    4    4    6  162   60]
     [ 218  109  131  116  305 2339]]


<font size = 3 >
    
 <p>
    
&nbsp; We can see that our best model is the random forrest , however we know there is something suspicious, our model's accuracy migh be pretty good but this is only happening because there is class imbalance and the models only predict the majority class, let's take a better look...

</p>
   
   
   <br>


```python
display(y_test_iphone.value_counts())
```


    5    2500
    0     615
    4     492
    3     399
    2     162
    1     114
    Name: iphonesentiment, dtype: int64


<font size = 3 >
    
 <p>
&nbsp;Well it looks like class 5 has is 58% of the test data, this means that basically our model is just predcting 5 because it's the most common level.This is a very common problem in classification called class imbalance, there are a few ways to deal with this issue 
    </p>
<ul>
   <li>Collect more data</li> 
  <li>Undersampling</li>
  <li>Oversampling</li>
  <li>Class recode (feature engineering)</li>
</ul> 
   
   
   <br>

<font size = 3 >
    
 <p>
    
&nbsp; To not make this any longer than necesary I'm gonna just give an overview of my approach to undersampling, the problems I ran into, how I fixed , and finnaly why I just gave up on it 
</p>
   
   
   <br>


```python
y_test_iphone.value_counts()
```




    5    2500
    0     615
    4     492
    3     399
    2     162
    1     114
    Name: iphonesentiment, dtype: int64



<br>



```python
minority = len(iphone_df[iphone_df['iphonesentiment'] == 1])

print('Minority class is 1 and has {} samples '.format(minority))
```

    Minority class is 1 and has 390 samples 



```python
dic = { 0: 0, 1:0, 2:0, 3:0, 4:0, 5:0 }

for i in range(6):
    dic[i]= iphone_df[iphone_df['iphonesentiment'] == i].index
    print(dic[i])
    
```

    Int64Index([    0,     1,     2,     3,     4,     7,     8,     9,    11,
                   12,
                ...
                12867, 12871, 12885, 12906, 12914, 12919, 12923, 12934, 12950,
                12969],
               dtype='int64', length=1962)
    Int64Index([ 824,  825,  827,  828,  829,  830,  831,  832,  833,  834,
                ...
                1297, 1298, 1299, 1300, 1301, 1302, 1303, 1304, 1306, 1308],
               dtype='int64', length=390)
    Int64Index([  103,   154,   310,   333,   450,   550,   591,   666,   884,
                 1028,
                ...
                10926, 11099, 11132, 11397, 11711, 11887, 11973, 12000, 12442,
                12726],
               dtype='int64', length=454)
    Int64Index([   10,    71,    74,   109,   138,   146,   149,   218,   258,
                  264,
                ...
                12750, 12751, 12752, 12757, 12797, 12824, 12887, 12895, 12918,
                12941],
               dtype='int64', length=1188)
    Int64Index([    5,     6,    77,   126,   143,   156,   194,   207,   212,
                  214,
                ...
                12737, 12806, 12813, 12831, 12849, 12879, 12896, 12904, 12956,
                12963],
               dtype='int64', length=1439)
    Int64Index([ 3423,  3425,  3426,  3427,  3428,  3429,  3430,  3431,  3432,
                 3433,
                ...
                12961, 12962, 12964, 12965, 12966, 12967, 12968, 12970, 12971,
                12972],
               dtype='int64', length=7540)



```python
dic_ind = { 0: 0, 1:0, 2:0, 3:0, 4:0, 5:0 }

for i in range(6):
    dic_ind[i] = np.random.choice(dic[i],minority,replace=False)
    
```


```python
 under_sample_indices = np.concatenate((dic_ind[0],dic_ind[1],dic_ind[2],dic_ind[3],dic_ind[4],dic_ind[5]), axis=0,  )
```


```python
under_sample_df = iphone_df.loc[under_sample_indices]

display(under_sample_df.head())

display(pd.DataFrame(under_sample_df['iphonesentiment'].tail()))


display(pd.DataFrame(under_sample_df['iphonesentiment'].value_counts()))

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
      <th>iphone</th>
      <th>samsunggalaxy</th>
      <th>sonyxperia</th>
      <th>nokialumina</th>
      <th>htcphone</th>
      <th>ios</th>
      <th>googleandroid</th>
      <th>iphonecampos</th>
      <th>samsungcampos</th>
      <th>sonycampos</th>
      <th>...</th>
      <th>sonyperunc</th>
      <th>nokiaperunc</th>
      <th>htcperunc</th>
      <th>iosperpos</th>
      <th>googleperpos</th>
      <th>iosperneg</th>
      <th>googleperneg</th>
      <th>iosperunc</th>
      <th>googleperunc</th>
      <th>iphonesentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>739</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>6763</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>322</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>27</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>6849</th>
      <td>11</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 59 columns</p>
</div>



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
      <th>iphonesentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9014</th>
      <td>5</td>
    </tr>
    <tr>
      <th>11581</th>
      <td>5</td>
    </tr>
    <tr>
      <th>6211</th>
      <td>5</td>
    </tr>
    <tr>
      <th>10082</th>
      <td>5</td>
    </tr>
    <tr>
      <th>3815</th>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>iphonesentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>390</td>
    </tr>
    <tr>
      <th>3</th>
      <td>390</td>
    </tr>
    <tr>
      <th>1</th>
      <td>390</td>
    </tr>
    <tr>
      <th>4</th>
      <td>390</td>
    </tr>
    <tr>
      <th>2</th>
      <td>390</td>
    </tr>
    <tr>
      <th>0</th>
      <td>390</td>
    </tr>
  </tbody>
</table>
</div>


<font size = 3 >
    
 <p>
    
&nbsp; This was a little bit obvious that it wouldnt work, my first idea was to train with this undersampled set and try to predict the the previous test_set I had   
</p>
   
   
   <br>


```python
X_iphone_us,y_iphone_us = X_Y_split(under_sample_df,'iphonesentiment')



display(X_iphone_us.head())

display(pd.DataFrame(y_iphone_us.head()))
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
      <th>iphone</th>
      <th>samsunggalaxy</th>
      <th>sonyxperia</th>
      <th>nokialumina</th>
      <th>htcphone</th>
      <th>ios</th>
      <th>googleandroid</th>
      <th>iphonecampos</th>
      <th>samsungcampos</th>
      <th>sonycampos</th>
      <th>...</th>
      <th>samsungperunc</th>
      <th>sonyperunc</th>
      <th>nokiaperunc</th>
      <th>htcperunc</th>
      <th>iosperpos</th>
      <th>googleperpos</th>
      <th>iosperneg</th>
      <th>googleperneg</th>
      <th>iosperunc</th>
      <th>googleperunc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>739</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>6763</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>322</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>27</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>6849</th>
      <td>11</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>



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
      <th>iphonesentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>739</th>
      <td>0</td>
    </tr>
    <tr>
      <th>6763</th>
      <td>0</td>
    </tr>
    <tr>
      <th>322</th>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0</td>
    </tr>
    <tr>
      <th>6849</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
start_time = time.time()

rf = RandomForestClassifier()

rf.fit(X_iphone_us,y_iphone_us)

predictions = rf.predict(X_test_iphone)

end_time = start_time - time.time()

print("--- Run Time for Random Forrest  is : %s mins--- \n " % np.round(((time.time()-start_time)/60),2))
print("Accuracy : ",accuracy_score(predictions,y_test_iphone))
print("\n")

print("Kappa's : ",cohen_kappa_score(predictions,y_test_iphone))

print("\n")
print("Confusion Matrix : \n \n  ",confusion_matrix(predictions,y_test_iphone) )

```

    --- Run Time for Random Forrest  is : 0.01 mins--- 
     
    Accuracy :  0.30009341429238673
    
    
    Kappa's :  0.22716364632980612
    
    
    Confusion Matrix : 
     
       [[ 398    0    0    4    7   26]
     [ 168  104   98   86  226 1817]
     [  15    3   53    7   21  164]
     [   7    0    0  281   10   58]
     [  12    2    3   12  199  185]
     [  15    5    8    9   29  250]]


<font size = 3 >
    
 <p>
    
&nbsp; Ofcourse this wouldn't work, you are training with 390 samples and trying to predict almost 5000, the samples were also chosent at random and here we can see that the model underperforms by a large margin.<br> <br>
&nbsp; I also tried other approaches such as cutting the examples of training of class 5 to a smaller number like 1500 and around those values, in order for a more balanced class distribution.<br> <br>
&nbsp; Nothing worked apparently, even with all the pre-processing the base model always outperfomed every new approach. <br> <br>
&nbsp; I finnaly turned back to feature engineering I tought if since I have such imbalance of classes maybe if I shrink the classes to a more optimal scale like 1- bad , 2- decent , 3- very good , my kappa and accuracy would increase, the art is in choosing the right balance  


</p>
   
   
   <br>


```python
iphone_df['iphonesentiment'] =  iphone_df['iphonesentiment'].map({0:1 , 1:1 , 2:2 , 3:2, 4:3, 5:3})
```


```python
iphone_df['iphonesentiment'].value_counts()
```




    3    8979
    1    2352
    2    1642
    Name: iphonesentiment, dtype: int64




```python
X_iphone,y_iphone = X_Y_split(iphone_df,'iphonesentiment')

X_train_iphone , X_test_iphone , y_train_iphone , y_test_iphone =  train_test_split(X_iphone, y_iphone, test_size=0.33, random_state=42)



```


```python
print('Test')
display(y_test_iphone.value_counts())

print("\n")
print("Train")
display(y_train_iphone.value_counts())
```

    Test



    3    2992
    1     729
    2     561
    Name: iphonesentiment, dtype: int64


    
    
    Train



    3    5987
    1    1623
    2    1081
    Name: iphonesentiment, dtype: int64



```python

start_time = time.time()

rf = RandomForestClassifier()

rf.fit(X_train_iphone,y_train_iphone)

predictions = rf.predict(X_test_iphone)

end_time = start_time - time.time()

print("--- Run Time is : %s mins--- \n " % np.round(((time.time()-start_time)/60),2))
print("Accuracy : ",accuracy_score(predictions,y_test_iphone))
print("\n")

print("Kappa's : ",cohen_kappa_score(predictions,y_test_iphone))

print("\n")
print("Confusion Matrix : \n \n  ",confusion_matrix(predictions,y_test_iphone) )

```

    --- Run Time is : 0.01 mins--- 
     
    Accuracy :  0.8493694535263895
    
    
    Kappa's :  0.61744757839659
    
    
    Confusion Matrix : 
     
       [[ 387    4   20]
     [   3  295   17]
     [ 339  262 2955]]


<font size = 3 >
    
 <p>
    
&nbsp; We finnaly got some improvements, I feel that a combination of both methods (undersampling) and featute engineering would be the most powerfull approach, however time is not infinite and trying to figure out python libraries everytime a new project arrives takes some time, I also designed this report this format so that future students don't loose so much time researching as I did.<br> <br>
&nbsp;Hopefully this is of some help !
</p>
   
   
   <br>

<font size = 3>
<div class="alert alert-warning">
<b>NOTE</b> I only performed the analysis for the iphone sentiment, the analysis for the galaxy is  analogous to the iphone, maybe with some performance differences      
</div>

<html> 
       <font size = 5, color = green >
            
<h1><center> Pre-processing + feature engineering   </center></h1>
        
   </font>

<html> 
       <font size = 3, color = green >
            
<h1><center> Correlation   </center></h1>
        
   </font>


```python
iphone_df_corr =correlation(iphone_df,0.855)


```


```python
X_iphone_corr,y_iphone_corr = X_Y_split(iphone_df_corr,'iphonesentiment')

X_train_iphone_corr , X_test_iphone_corr , y_train_iphone_corr , y_test_iphone_corr =  train_test_split(X_iphone_corr, y_iphone_corr, test_size=0.33, random_state=42)



```


```python

start_time = time.time()

rf = RandomForestClassifier()

rf.fit(X_train_iphone_corr,y_train_iphone_corr)

predictions = rf.predict(X_test_iphone_corr)

end_time = start_time - time.time()

print("--- Run Time is : %s mins--- \n " % np.round(((time.time()-start_time)/60),2))
print("Accuracy : ",accuracy_score(predictions,y_test_iphone_corr))
print("\n")

print("Kappa's : ",cohen_kappa_score(predictions,y_test_iphone_corr))

print("\n")
print("Confusion Matrix : \n \n  ",confusion_matrix(predictions,y_test_iphone_corr) )

```

    --- Run Time is : 0.01 mins--- 
     
    Accuracy :  0.8365249883232134
    
    
    Kappa's :  0.5819583523184269
    
    
    Confusion Matrix : 
     
       [[ 385    6   11]
     [   7  256   40]
     [ 337  299 2941]]


<html> 
       <font size = 3, color = green >
            
<h1><center> Near-zero var imp  </center></h1>
        
   </font>


```python
iphone_df_near =  near_zero_var(iphone_df,0.5)

```


```python
X_iphone_near,y_iphone_near = X_Y_split(iphone_df_near,'iphonesentiment')

X_train_iphone_near , X_test_iphone_near , y_train_iphone_near , y_test_iphone_near =  train_test_split(X_iphone_near, y_iphone_near, test_size=0.33, random_state=42)



```


```python

start_time = time.time()

rf = RandomForestClassifier()

rf.fit(X_train_iphone_near,y_train_iphone_near)

predictions = rf.predict(X_test_iphone_near)

end_time = start_time - time.time()

print("--- Run Time is : %s mins--- \n " % np.round(((time.time()-start_time)/60),2))
print("Accuracy : ",accuracy_score(predictions,y_test_iphone_near))
print("\n")

print("Kappa's : ",cohen_kappa_score(predictions,y_test_iphone_near))

print("\n")
print("Confusion Matrix : \n \n  ",confusion_matrix(predictions,y_test_iphone_near) )

```

    --- Run Time is : 0.01 mins--- 
     
    Accuracy :  0.8176085941148996
    
    
    Kappa's :  0.5169585962076194
    
    
    Confusion Matrix : 
     
       [[ 382   26   16]
     [   2  161   18]
     [ 345  374 2958]]


<html> 
       <font size = 3, color = green >
            
<h1><center> RFCEV   </center></h1>
        
   </font>


```python
iphone_imp = most_imp_vars(iphone_df,varimpihpone)
```

    htccamunc
    iphonecamneg
    nokiadisunc
    sonycampos
    samsungdisneg
    iphonecampos
    samsungperpos
    samsungcamunc
    nokiacamunc
    sonydisneg
    iphoneperpos
    nokiadispos
    nokiacampos
    samsungdisunc
    nokiacamneg
    iphoneperneg
    iphonedisneg
    htcperneg
    ios
    nokiadisneg



```python
X_iphone_imp,y_iphone_imp = X_Y_split(iphone_imp,'iphonesentiment')

X_train_iphone_imp , X_test_iphone_imp , y_train_iphone_imp , y_test_iphone_imp =  train_test_split(X_iphone_imp, y_iphone_imp, test_size=0.33, random_state=42)



```


```python

start_time = time.time()

rf = RandomForestClassifier()

rf.fit(X_train_iphone_imp,y_train_iphone_imp)

predictions = rf.predict(X_test_iphone_imp)

end_time = start_time - time.time()

print("--- Run Time is : %s mins--- \n " % np.round(((time.time()-start_time)/60),2))
print("Accuracy : ",accuracy_score(predictions,y_test_iphone_imp))
print("\n")

print("Kappa's : ",cohen_kappa_score(predictions,y_test_iphone_imp))

print("\n")
print("Confusion Matrix : \n \n  ",confusion_matrix(predictions,y_test_iphone_imp) )

```

    --- Run Time is : 0.01 mins--- 
     
    Accuracy :  0.8278841662774404
    
    
    Kappa's :  0.5548816368579648
    
    
    Confusion Matrix : 
     
       [[ 386    5   11]
     [   8  220   42]
     [ 335  336 2939]]


<html> 
       <font size = 3 >
            
&nbsp;As I said previously pre processing seems to only have negative impact on the models performance, maybe some more autotuning on the models would probably improve slighly the performance.<br> <br>


&nbsp;I would reccomend trying different parameters , one specifically called class weights, which assigns differetn weights for different classes in order to fix the class imbalance issue, there are other options maybe a combination of undersampling with oversampling recoding and prepocessing you would reach an optimal <br> <br>

&nbsp; If anyone finds an elegant solution more efficient solution, please send it to me  I would absolutely love to take a look and learn from it 
        
   </font>

<html> 
       <font size = 5, color = green >
            
<h1><center> Clean Data    </center></h1>
        
   </font>


```python
samsung_df = pd.read_csv(r"galaxy_smallmatrix_labeled_9d.csv")
iphone_df = pd.read_csv(r"iphone_smallmatrix_labeled_9d.csv")
```


```python
iphone_df['iphone'].var()
```




    32.08345239866406




```python
iphone_df[iphone_df.iphone > iphone_df.iphone.median() ]
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
      <th>iphone</th>
      <th>samsunggalaxy</th>
      <th>sonyxperia</th>
      <th>nokialumina</th>
      <th>htcphone</th>
      <th>ios</th>
      <th>googleandroid</th>
      <th>iphonecampos</th>
      <th>samsungcampos</th>
      <th>sonycampos</th>
      <th>...</th>
      <th>sonyperunc</th>
      <th>nokiaperunc</th>
      <th>htcperunc</th>
      <th>iosperpos</th>
      <th>googleperpos</th>
      <th>iosperneg</th>
      <th>googleperneg</th>
      <th>iosperunc</th>
      <th>googleperunc</th>
      <th>iphonesentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>41</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>24</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>25</th>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>29</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>32</th>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>38</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>41</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>46</th>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>55</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>67</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>75</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>82</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>83</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>95</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>101</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>108</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>111</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>118</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>120</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>122</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>125</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>126</th>
      <td>41</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>128</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>141</th>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>143</th>
      <td>41</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>158</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>160</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>166</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>12825</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12828</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12831</th>
      <td>41</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>12833</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12835</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12841</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12843</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12848</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12849</th>
      <td>41</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>12850</th>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12856</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12859</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12871</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>12873</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12879</th>
      <td>41</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>12883</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12899</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12903</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12904</th>
      <td>41</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>12905</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12914</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>12917</th>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12924</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12930</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12937</th>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12950</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>12956</th>
      <td>41</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>12957</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12969</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>12971</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>2852 rows × 59 columns</p>
</div>




```python

```
