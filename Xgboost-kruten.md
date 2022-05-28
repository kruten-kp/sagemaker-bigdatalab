1. Importing necessary Libraries
2. Creating S3 bucket 
3. Mapping train And Test Data in S3
4. Mapping The path of the models in S3


```python
import sagemaker
import boto3
from sagemaker.amazon.amazon_estimator import get_image_uri 
from sagemaker.session import s3_input, Session
```


```python
bucket_name = '3bankapplication3' 
my_region = boto3.session.Session().region_name 
print(my_region)
```

    us-east-1



```python
s3 = boto3.resource('s3')
try:
    if  my_region == 'us-east-1':
        s3.create_bucket(Bucket=bucket_name)
    print('S3 bucket created successfully')
except Exception as e:
    print('S3 error: ',e)
```

    S3 bucket created successfully



```python

prefix = 'xgboost-as-a-built-in-algo'
output_path ='s3://{}/{}/output'.format(bucket_name, prefix)
print(output_path)
```

    s3://3bankapplication3/xgboost-as-a-built-in-algo/output


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



```python

import os
pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], 
                                                axis=1)], 
                                                axis=1).to_csv('train.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')
```


```python
# Test Data Into Buckets
pd.concat([test_data['y_yes'], test_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('test.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'test/test.csv')).upload_file('test.csv')
s3_input_test = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/test'.format(bucket_name, prefix), content_type='csv')
```

### Building Models Xgboot- Inbuilt Algorithm


```python

container = sagemaker.image_uris.retrieve('xgboost', boto3.Session().region_name, version = '1.3-1')
```


```python

hyperparameters = {
        "max_depth":"5",
        "eta":"0.2",
        "gamma":"4",
        "min_child_weight":"6",
        "subsample":"0.7",
        "objective":"binary:logistic",
        "num_round":50
        }
```


```python

estimator = sagemaker.estimator.Estimator(image_uri=container, 
                                          hyperparameters=hyperparameters,
                                          role=sagemaker.get_execution_role(),
                                          instance_count=1, 
                                          instance_type='ml.m5.2xlarge', 
                                          volume_size=5, # 5 GB 
                                          output_path=output_path,
                                          use_spot_instances=True,
                                          max_run=300,
                                          max_wait=600)
```


```python
estimator.fit({'train': s3_input_train,'validation': s3_input_test})
```

    2022-05-28 02:46:02 Starting - Starting the training job...
    2022-05-28 02:46:04 Starting - Launching requested ML instancesProfilerReport-1653705962: InProgress
    .........
    2022-05-28 02:47:53 Starting - Preparing the instances for training.........
    2022-05-28 02:49:31 Downloading - Downloading input data
    2022-05-28 02:49:31 Training - Training image download completed. Training in progress.
    2022-05-28 02:49:31 Uploading - Uploading generated training model.[34m[2022-05-28 02:49:28.260 ip-10-0-184-19.ec2.internal:1 INFO utils.py:27] RULE_JOB_STOP_SIGNAL_FILENAME: None[0m
    [34m[2022-05-28:02:49:28:INFO] Imported framework sagemaker_xgboost_container.training[0m
    [34m[2022-05-28:02:49:28:INFO] Failed to parse hyperparameter objective value binary:logistic to Json.[0m
    [34mReturning the value itself[0m
    [34m[2022-05-28:02:49:28:INFO] No GPUs detected (normal if no gpus installed)[0m
    [34m[2022-05-28:02:49:28:INFO] Running XGBoost Sagemaker in algorithm mode[0m
    [34m[2022-05-28:02:49:28:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2022-05-28:02:49:28:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2022-05-28:02:49:28:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2022-05-28:02:49:28:INFO] Determined delimiter of CSV input is ','[0m
    [34m[2022-05-28:02:49:28:INFO] Single node training.[0m
    [34m[2022-05-28:02:49:28:INFO] Train matrix has 28831 rows and 59 columns[0m
    [34m[2022-05-28:02:49:28:INFO] Validation matrix has 12357 rows[0m
    [34m[02:49:28] WARNING: ../src/learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.[0m
    [34m[0]#011train-logloss:0.57285#011validation-logloss:0.57388[0m
    [34m[1]#011train-logloss:0.49296#011validation-logloss:0.49483[0m
    [34m[2]#011train-logloss:0.43699#011validation-logloss:0.44017[0m
    [34m[3]#011train-logloss:0.39663#011validation-logloss:0.40048[0m
    [34m[4]#011train-logloss:0.36733#011validation-logloss:0.37209[0m
    [34m[5]#011train-logloss:0.34574#011validation-logloss:0.35146[0m
    [34m[6]#011train-logloss:0.32987#011validation-logloss:0.33640[0m
    [34m[7]#011train-logloss:0.31788#011validation-logloss:0.32478[0m
    [34m[8]#011train-logloss:0.30915#011validation-logloss:0.31662[0m
    [34m[9]#011train-logloss:0.30226#011validation-logloss:0.31044[0m
    [34m[10]#011train-logloss:0.29718#011validation-logloss:0.30609[0m
    [34m[11]#011train-logloss:0.29316#011validation-logloss:0.30271[0m
    [34m[12]#011train-logloss:0.29028#011validation-logloss:0.30029[0m
    [34m[13]#011train-logloss:0.28766#011validation-logloss:0.29825[0m
    [34m[14]#011train-logloss:0.28567#011validation-logloss:0.29661[0m
    [34m[15]#011train-logloss:0.28434#011validation-logloss:0.29565[0m
    [34m[16]#011train-logloss:0.28328#011validation-logloss:0.29466[0m
    [34m[17]#011train-logloss:0.28189#011validation-logloss:0.29366[0m
    [34m[18]#011train-logloss:0.28103#011validation-logloss:0.29344[0m
    [34m[19]#011train-logloss:0.28020#011validation-logloss:0.29296[0m
    [34m[20]#011train-logloss:0.27964#011validation-logloss:0.29269[0m
    [34m[21]#011train-logloss:0.27923#011validation-logloss:0.29267[0m
    [34m[22]#011train-logloss:0.27900#011validation-logloss:0.29280[0m
    [34m[23]#011train-logloss:0.27840#011validation-logloss:0.29243[0m
    [34m[24]#011train-logloss:0.27799#011validation-logloss:0.29234[0m
    [34m[25]#011train-logloss:0.27755#011validation-logloss:0.29206[0m
    [34m[26]#011train-logloss:0.27718#011validation-logloss:0.29196[0m
    [34m[27]#011train-logloss:0.27688#011validation-logloss:0.29192[0m
    [34m[28]#011train-logloss:0.27663#011validation-logloss:0.29183[0m
    [34m[29]#011train-logloss:0.27642#011validation-logloss:0.29160[0m
    [34m[30]#011train-logloss:0.27613#011validation-logloss:0.29140[0m
    [34m[31]#011train-logloss:0.27585#011validation-logloss:0.29134[0m
    [34m[32]#011train-logloss:0.27554#011validation-logloss:0.29136[0m
    [34m[33]#011train-logloss:0.27516#011validation-logloss:0.29135[0m
    [34m[34]#011train-logloss:0.27511#011validation-logloss:0.29132[0m
    [34m[35]#011train-logloss:0.27494#011validation-logloss:0.29133[0m
    [34m[36]#011train-logloss:0.27468#011validation-logloss:0.29143[0m
    [34m[37]#011train-logloss:0.27429#011validation-logloss:0.29122[0m
    [34m[38]#011train-logloss:0.27402#011validation-logloss:0.29117[0m
    [34m[39]#011train-logloss:0.27385#011validation-logloss:0.29128[0m
    [34m[40]#011train-logloss:0.27366#011validation-logloss:0.29118[0m
    [34m[41]#011train-logloss:0.27352#011validation-logloss:0.29117[0m
    [34m[42]#011train-logloss:0.27316#011validation-logloss:0.29111[0m
    [34m[43]#011train-logloss:0.27294#011validation-logloss:0.29095[0m
    [34m[44]#011train-logloss:0.27274#011validation-logloss:0.29101[0m
    [34m[45]#011train-logloss:0.27254#011validation-logloss:0.29093[0m
    [34m[46]#011train-logloss:0.27240#011validation-logloss:0.29085[0m
    [34m[47]#011train-logloss:0.27223#011validation-logloss:0.29084[0m
    [34m[48]#011train-logloss:0.27210#011validation-logloss:0.29078[0m
    [34m[49]#011train-logloss:0.27198#011validation-logloss:0.29069[0m
    
    2022-05-28 02:49:54 Completed - Training job completed
    Training seconds: 33
    Billable seconds: 13
    Managed Spot Training savings: 60.6%


### Deploy Machine Learning Model As Endpoints


```python
xgb_predictor = estimator.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge')
```

    ---------!

#### Prediction of the Test Data


```python
test_data_array.shape
```




    (12357, 59)




```python
from sagemaker.predictor import csv_serializer
test_data_array = test_data.drop(['y_no', 'y_yes'], axis=1).values #load the data into an array
# xgb_predictor.content_type = 'text/csv' # set the data type for an inference
xgb_predictor.__dict__.keys()
xgb_predictor.serializer = csv_serializer # set the serializer type
predictions = xgb_predictor.predict(test_data_array).decode('utf-8') # predict!
predictions_array = np.fromstring(predictions[1:], sep='\n') # and turn the prediction into an array
print(predictions_array.shape);
```

    The csv_serializer has been renamed in sagemaker>=2.
    See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.


    (12357,)



```python
predictions_array
```




    array([0.05214286, 0.05660191, 0.05096195, ..., 0.03436061, 0.02942475,
           0.03715819])




```python
cm = pd.crosstab(index=test_data['y_yes'], columns=np.round(predictions_array), rownames=['Observed'], colnames=['Predicted'])
tn = cm.iloc[0,0]; fn = cm.iloc[1,0]; tp = cm.iloc[1,1]; fp = cm.iloc[0,1]; p = (tp+tn)/(tp+tn+fp+fn)*100
print("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
print("{0:<15}{1:<15}{2:>8}".format("Predicted", "No Purchase", "Purchase"))
print("Observed")
print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format("No Purchase", tn/(tn+fn)*100,tn, fp/(tp+fp)*100, fp))
print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format("Purchase", fn/(tn+fn)*100,fn, tp/(tp+fp)*100, tp))
```

    
    Overall Classification Rate: 89.7%
    
    Predicted      No Purchase    Purchase
    Observed
    No Purchase    91% (10785)    34% (151)
    Purchase        9% (1124)     66% (297) 
    



```python

```

## Endpoint of the Xgboost trained model


```python
sm_boto3 = boto3.client("sagemaker")

endpoint_name = 'wang-xgboost'
ep_des_res = sm_boto3.describe_endpoint(EndpointName=endpoint_name)

print(ep_des_res);
```

    {'EndpointName': 'wang-xgboost', 'EndpointArn': 'arn:aws:sagemaker:us-east-1:323466860379:endpoint/wang-xgboost', 'EndpointConfigName': 'lucky-banaya', 'ProductionVariants': [{'VariantName': 'default-variant-name', 'DeployedImages': [{'SpecifiedImage': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.3-1', 'ResolvedImage': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost@sha256:f76a3ade03c5a8813d40d70041ac166dc66e0e68627ba0995d47552a599cf6dc', 'ResolutionTime': datetime.datetime(2022, 5, 28, 8, 31, 24, 226000, tzinfo=tzlocal())}], 'CurrentWeight': 1.0, 'DesiredWeight': 1.0, 'CurrentInstanceCount': 1, 'DesiredInstanceCount': 1}], 'EndpointStatus': 'InService', 'CreationTime': datetime.datetime(2022, 5, 28, 8, 31, 23, 537000, tzinfo=tzlocal()), 'LastModifiedTime': datetime.datetime(2022, 5, 28, 8, 35, 33, 558000, tzinfo=tzlocal()), 'AsyncInferenceConfig': {'OutputConfig': {'S3OutputPath': 's3://model--outputs/wang-xgboost/'}}, 'ResponseMetadata': {'RequestId': '81ed6503-1ea5-4e3f-8f03-5959307ff719', 'HTTPStatusCode': 200, 'HTTPHeaders': {'x-amzn-requestid': '81ed6503-1ea5-4e3f-8f03-5959307ff719', 'content-type': 'application/x-amz-json-1.1', 'content-length': '784', 'date': 'Sat, 28 May 2022 08:56:00 GMT'}, 'RetryAttempts': 0}}



```python

```


```python

```

#### Deleting The Endpoints


```python
# sagemaker.Session().delete_endpoint(xgb_predictor.endpoint)
# bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
# bucket_to_delete.objects.all().delete()
```


```python

```
