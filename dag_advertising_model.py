# !pip3 install xgboost

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import pandas as pd
from google.cloud import storage
import gcsfs,json
from datetime import datetime
from google.cloud import bigquery
from google.cloud import logging
from imblearn.over_sampling import RandomOverSampler

from advertising_model_training import (
    read_campaign_data, calculate_revenue_per_month, calculate_spend_per_month,merge_dataframes,
    write_metrics_to_bigquery, train_model,evaluate_model,save_model
)

logging_client = logging.Client()
logger = logging_client.logger('Advertising_model-logs')

def validate_csv():
    # Load data
    fs = gcsfs.GCSFileSystem()

    campaign_file_path = "gs://assignment_advertising_model/advertising_roi/campaign_spend.csv"
    df_spend = pd.read_csv(campaign_file_path)


    revenue_file_path = "gs://assignment_advertising_model/advertising_roi/monthly_revenue.csv"
    df_revenue_per_month = pd.read_csv(revenue_file_path)

    # Define expected columns
    spend_expected_cols = ['CAMPAIGN','CHANNEL','DATE','TOTAL_CLICKS','TOTAL_COST','ADS_SERVED']
    revenue_expected_cols =['YEAR','MONTH','REVENUE']
    
    # Check if the loaded columns are same as expected columns
    if (list(df_spend.columns) == spend_expected_cols) and (list(df_revenue_per_month.columns) == revenue_expected_cols):
        return True
    else:
        logger.log_struct({
            'keyword': 'Advertising_model_Training',
            'description': 'This log captures the last run for Model Training',
            'training_timestamp': datetime.now().isoformat(),
            'model_output_msg': "Input Data is not valid",
            'training_status':0
        })
        raise ValueError(f'CSV does not have expected columns. Columns in CSV are: {list(df.columns)}')

def read_last_training_metrics():
    client = bigquery.Client()
    table_id = "core-site-401904.ml_ops.advertising_roi_model_metrics"
    query = f"""
        SELECT *
        FROM `{table_id}`
        ORDER BY training_time DESC
        LIMIT 1
    """
    result = client.query(query).result()
    latest_row = next(result)
    return json.loads(latest_row[2])

def evaluate_model_dag():
    # Load data for evaluation
    campaign_file_path = "gs://assignment_advertising_model/advertising_roi/campaign_spend.csv"
    df_spend = read_campaign_data(campaign_file_path)

    revenue_file_path = "gs://assignment_advertising_model/advertising_roi/monthly_revenue.csv"
    df_revenue_per_month = calculate_revenue_per_month(revenue_file_path)

    df_spend_per_month = calculate_spend_per_month(df_spend)

    df_joined = merge_dataframes(df_revenue_per_month, df_spend_per_month)
   # print(df_joined.head())
    model, X_train, y_train, X_test, y_test = train_model(df_joined)
    train_r2_score, test_r2_score = evaluate_model(model, X_train, y_train, X_test, y_test)
    # Get the current model metrics for evaluation


    # Get the last/existing model metrics for comparison
    last_model_metrics = read_last_training_metrics()
    logger.log_struct({ 'last_model_metrics >>>>>>>>>>>log <<<<<<>>>>>':last_model_metrics})
    print('last_model_metrics >>>>>>>>>>><<<<<<>>>>>'+str(last_model_metrics))
    last_train_r2_score = last_model_metrics['r2_train']
    last_test_r2_score = last_model_metrics['r2_test']

    # Define the threshold values for train_r2_score and test_r2_score
    train_r2_score_threshold = 0.5
    test_r2_score_threshold = 0.5
    
    # Save the model artifact if metrics are above the thresholds
    if (train_r2_score >=train_r2_score_threshold and test_r2_score >=test_r2_score_threshold) :
   #and (train_r2_score >= last_train_r2_score and test_r2_score >= last_test_r2_score):
        save_model(model)
        
        model_metrics = {"r2_train":train_r2_score,"r2_test":test_r2_score}    
        training_time = datetime.now()
        model_name = "linear_regression"
        write_metrics_to_bigquery(model_name, training_time,model_metrics)
        logger.log_struct({
            'keyword': 'Advertising_model_Training',
            'description': 'This log captures the last run for Model Training',
            'training_timestamp': datetime.now().isoformat(),
            'model_output_msg': "Model artifact saved",
            'training_status':1
        })
    else:
        logger.log_struct({
            'keyword': 'Advertising_model_Training',
            'description': 'This log captures the last run for Model Training',
            'training_timestamp': datetime.now().isoformat(),
            'model_output_msg': "Model metrics do not meet the defined threshold",
            'model_metrics':model_metrics,
            'training_status':0
        })

# Define the default_args dictionary
default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    'retries': 1,
}

# Instantiate the DAG
dag = DAG(
    'Advertising_model_Training',
    default_args=default_args,
    description='My Assignment',
    schedule_interval=None,
)

# Define the tasks/operators
validate_csv_task = PythonOperator(
    task_id='validate_csv',
    python_callable=validate_csv,
    dag=dag,
)

evaluation_task = PythonOperator(
    task_id='model_evaluation1',
    python_callable=evaluate_model_dag,
    dag=dag,
)

validate_csv_task >> evaluation_task