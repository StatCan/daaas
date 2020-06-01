# MLflow for model tracking

MLflow is an open source platform for Machine Learning Lifecycle.MLflow also includes experimentation, reproducibility, deployement and Model Registry.
There are four important components in MLFlow to create Machine Learning Lifecycle.
#### 1. MLFlow Tracking
#### 2. MLFlow Projects
#### 3. MLFlow Models
#### 4. MLFlow Registry
## MLFlow Tracking
MlFlow Tracking component is an API and UI for logging parameters, metrics and output files when running the machine learning code. These logged parameters, metrics and output files can be used to visualize the results. In MLFlow one can log and query using Python, Rest API, R and Java API APIs.
## How to track a Machine Learning Model using MLflow?
Here we will be using Python Jupyter Notebook to create an example and track the model.
Mlflow can be installed by using the below command.

![Install MLFlow](../images/installmlflow.PNG)

To track the model we need to import MLFlow in jupyter notebook and start an experiment. To start a new experiment we need to create a new experiment and set the experiment path to a path where we want to save our experiment files.

![Import MLflow and create a new experiment](../images/mlflow_createexperiment.PNG)

Now that we have our experiment set we need to start thinking about what do we want to save in our experiment.

**Let's get started!**

MLflow categorizes into three main categories:

1.Parameters

2.Metrics

3.Artifacts

Apart from these three MLflow can also track Code version, Start and End time of the run and Source.

The first step to log parameters, metrics or artifacts is to start a Mlflow run. Mlflow run can be started using **mlflow.start_run(run_name = "testrun")** command.
All the logged parameters, metrics and artifacts will saved under the current run_name. If you want to save it in a different run, you have to end the current run using **mlflow.end_run(run_name = "testrun")** and then start a new run to log new values.

If you run your notebook then the runs will get saved in a working directory in a newly created **./mlruns**

#### How to log parameters in MLflow?

Parameters are variables that you can change and tweak in your Machine Learning Models. These parameters can be logged using **mlflow.log_param()**.

![Logging Parameters](../images/paramlog_mlflow.PNG)

As seen in the example the logged parameters can be seen in mlruns folder under param subfolder. You can log multiple parameters under the same runname.

#### How to log metrics in MLflow?

Metrics are values that you want to measure as a result of tweaking your parameters. Typical metrics that are tracked can be items like F1 score, RMSE, MAE, AUC etc. These metrics can be logged using **mlflow.log_metric()**

Just like the parameters the metrics can also be found in the same mlflow run directory under metric folder.

To inspect the logged data in a very user-friendly manner, stay in the same directory and run **mlflow ui** in the command line.

Then you can access the http link provided and the mlflow UI home page will be displayed. You can find logged metrics and parameters under the experiment name which you have saved in your notebook

![MLflow UI](../images/MLflow-UI.PNG)

#### How to log artifacts in MLflow?

Artifacts can keep track of graphs, any other important variables. Artifacts can also keep track of saved models. From Mlflow UI the artifacts can also be downloaded to save for future reference.

Artifacts can be saved using **mlflow.log_artifact(name of the graph)**

The saved artifact can be seen in artifacts folder from mlruns and also from artifacts in MLflow UI.

![Logged Artifacts](../images/Artifacts_mlflow.PNG)

#### How to save model in MLflow?

ML flow allows you to save the model that is being trained. Models can be saved in artifacts using **mlflow.log_model(model name)**

or Models can be directly saved to local directory using **mlflow.save_model(model name)**

![Saved_Model](../images/Save_model.PNG)






