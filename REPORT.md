> Name: **Kruten Patel**
> ID: **[GM76254]**


# Abstract:

The project takes a Bank Application dataset which loads the dataset in the S3 bucket, saves the train and test dataset in the buckets and then builds XGBoost containers. A sagemaker is constructed to call the containers previously created. This helps to implement Machine Learning models as Endpoints. 

# Introduction:

The Bank Application Dataset contains the customer details, such as their working details, profession, education history, housing details, loan application details, age, gender and along with this information the dataset also includes the probability of the customer purchasing the product in the binary format whether the customer is going to purchase the product then the percentage accuracy of the model is represented over here. In the first few steps we download the dataset and store it in S3 as depicted below.

### Downloading The Dataset And Storing in S3


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

# Advantages of XGBoost Algorithm over other Machine Learning Algorithms:

In general XGBoost Algorithm makes the working efficient by providing features such as  Regularization with in-built L1 and L2 Lasso regression. It helps solve any discrepancies with overfitting models. Secondly allows users to run a cross-validation at each iteration of the boosting process and thus it is easy to get the exact optimum number of boosting iterations in a single run.

When XGBoost encounters a missing value at a node, it tries both the left and right hand split and learns the way leading to higher loss for each node. It then does the same when working on the testing data. In the project  an output path where the trained model is created before the data is split into train and test dataset. The train and the test data is split into 70-30. 

XGBoost allows the user to perform cross-validation at each iteration of the boosting process, making it simple to obtain the correct number of boosting iterations in a single run.
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

# Deploying the model:
So we were able to clean our data and then use that data to build a model that can be used to make predictions. But, for the time being, we don't want to have to go into Sagemaker and restart the notebook every time. Furthermore, we want to make the predictions available to a web application in the future.
Sagemaker allows you to create an endpoint to which you can send data and receive results based on your model.


### Deploy Machine Learning Model As Endpoints


```python
xgb_predictor = estimator.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge')
```

    ---------!


# Prediction

### Prediction of the Test Data


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
    


# Challenges faced and solution:
There was a lot of time taken by the model inorder to get executed and sometimes due to the changing versions of AWS Sagemaker the code which was written in the past also had to be updated according to the latest version of the code. Inorder to seriously reduce the time interval we are making the use of API’s



# API Endpoint


```python
sm_boto3 = boto3.client("sagemaker")

endpoint_name = 'wang-xgboost'
ep_des_res = sm_boto3.describe_endpoint(EndpointName=endpoint_name)

print(ep_des_res);
```

    {'EndpointName': 'wang-xgboost', 'EndpointArn': 'arn:aws:sagemaker:us-east-1:323466860379:endpoint/wang-xgboost', 'EndpointConfigName': 'lucky-banaya', 'ProductionVariants': [{'VariantName': 'default-variant-name', 'DeployedImages': [{'SpecifiedImage': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.3-1', 'ResolvedImage': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost@sha256:f76a3ade03c5a8813d40d70041ac166dc66e0e68627ba0995d47552a599cf6dc', 'ResolutionTime': datetime.datetime(2022, 5, 28, 8, 31, 24, 226000, tzinfo=tzlocal())}], 'CurrentWeight': 1.0, 'DesiredWeight': 1.0, 'CurrentInstanceCount': 1, 'DesiredInstanceCount': 1}], 'EndpointStatus': 'InService', 'CreationTime': datetime.datetime(2022, 5, 28, 8, 31, 23, 537000, tzinfo=tzlocal()), 'LastModifiedTime': datetime.datetime(2022, 5, 28, 8, 35, 33, 558000, tzinfo=tzlocal()), 'AsyncInferenceConfig': {'OutputConfig': {'S3OutputPath': 's3://model--outputs/wang-xgboost/'}}, 'ResponseMetadata': {'RequestId': '81ed6503-1ea5-4e3f-8f03-5959307ff719', 'HTTPStatusCode': 200, 'HTTPHeaders': {'x-amzn-requestid': '81ed6503-1ea5-4e3f-8f03-5959307ff719', 'content-type': 'application/x-amz-json-1.1', 'content-length': '784', 'date': 'Sat, 28 May 2022 08:56:00 GMT'}, 'RetryAttempts': 0}}


# Result:
In the end we execute the prediction of the test data, which is the number of customers that are actually purchasing the product and the number of customers that are not going to purchase the product, we do it in the form of percentage prediction accuracy as shown below.
