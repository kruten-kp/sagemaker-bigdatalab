> Name: **Kruten Patel**
> ID: **[GM76254]**


# Abstract:

The project takes a Bank Application dataset which loads the dataset in the S3 bucket, saves the train and test dataset in the buckets and then builds XGBoost containers. A sagemaker is constructed to call the containers previously created. This helps to implement Machine Learning models as Endpoints. 


# Predictions

#### Downloading The Dataset And Storing in S3


```python
import pandas as pd
import urllib
try:
    urllib.request.urlretrieve ("https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv", "bank_clean.csv")
    print('Success: downloaded bank_clean.csv.')
except Exception as e:
    print('Data load error: ',e)

try:
    model_data = pd.read_csv('./bank_clean.csv',index_col=0)
    print('Success: Data loaded into dataframe.')
except Exception as e:
    print('Data load error: ',e)
```

    Success: downloaded bank_clean.csv.
    Success: Data loaded into dataframe.



```python
### Train Test split

import numpy as np
train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
print(train_data.shape, test_data.shape)
```

    (28831, 61) (12357, 61)




