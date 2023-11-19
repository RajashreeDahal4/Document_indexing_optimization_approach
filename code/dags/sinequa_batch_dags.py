from airflow import DAG
from airflow.decorators import task
from uuid import uuid4
from airflow.operators.empty import EmptyOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.python import PythonOperator

from airflow.utils.dates import days_ago
from dateutil.parser import parse
import os
import json
import torch
import requests
from time import sleep
import numpy as np

from datetime import timedelta
from airflow.utils.dates import datetime
from airflow.providers.amazon.aws.operators.s3 import S3ListOperator
from airflow_multi_dagrun.operators import TriggerMultiDagRunOperator
from time import sleep
from airflow.hooks.base_hook import BaseHook


doc_md_DAG = """
### <span style="color:blue">Airbus metadata conversion DAG</span> 
#### Airbus metadata conversion DAG:
This DAG converts airbus metadata into STAC. 
"""
def chunk_text_blobs(*kwargs):
    import boto3
    import json 
    with open("/opt/airflow/dags/config.json","rb") as file:
        config=json.load(file)
    bucket_name=config["aws_parameters"]["bucket_name"]
    aws_conn_id=config["aws_parameters"]["aws_conn_id"]
    folder_name=config["aws_parameters"]["bucket_folder_name"]
    aws_hook = BaseHook.get_hook(conn_id=aws_conn_id)
    # Retrieve AWS credentials
    aws_credentials = aws_hook.get_credentials()
    s3 = boto3.client('s3', aws_access_key_id=aws_credentials.access_key,
                            aws_secret_access_key=aws_credentials.secret_key,
                            aws_session_token=aws_credentials.token)
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix='sinequainputs/')
    object_name=[]
    
    for page in pages:
        for obj in page['Contents']:
            if ".json" in obj["Key"]:
                object_name.append(obj["Key"])
    all_files=[]
    print("The total files in the s3 bucket is",len(object_name))
    for i in object_name:
    # Read the JSON content from the S3 object
        response = s3.get_object(Bucket=bucket_name, Key=i)
        json_content = json.loads(response['Body'].read().decode('utf-8'))
        all_files.append(json_content)

    ids=[data["id"] for data in all_files]
    urls=[data['url'] for data in all_files]
    text_blobs=[data['text_blob'] for data in all_files]
    chunk_size=config["chunk_size"]
    url_chunks = [urls[i : i + chunk_size] for i in range(0, len(urls), chunk_size)]
    id_chunks=[ids[i : i + chunk_size] for i in range(0, len(ids), chunk_size)]
    text_blob_chunks=[text_blobs[i : i + chunk_size] for i in range(0, len(text_blobs), chunk_size)]
    for enum,i in enumerate(url_chunks):
        print("IT is:",{"data":{"urls":url_chunks[enum],"ids":id_chunks[enum],"text_blobs":text_blob_chunks[enum]}})
        yield {"data":{"ids":id_chunks[enum],"text_blobs":text_blob_chunks[enum],"urls":url_chunks[enum]}}
        sleep(1)
    for object_key in object_name:
        s3.delete_object(Bucket=bucket_name, Key=object_key)

with DAG(
    dag_id="sinequa_batch_inference",
    doc_md=doc_md_DAG,
    catchup=False,
    schedule=None,
    start_date=days_ago(0),
    params={"data":{"ids": [i for i in range(2)], "urls": [f"this is a test urls_{i}" for i in range(2)], "text_blobs": [f"This is a test blob{i}" for i in range(2)]}}) as dag:
    start = EmptyOperator(task_id="start", dag=dag)
    stop = EmptyOperator(task_id="stop", dag=dag)

    @task(max_active_tis_per_dag=100)
    def process_urls(**kwargs):
        from preprocessing import Preprocessor
        import pandas as pd
        from encoder import Encoder
        image_list,pdf_list=[],[]
        config = kwargs.get("dag_run").conf
        data=config['data']
        url_list=data['urls']
        id_lists=data['ids']
        text_blob_lists=data['text_blobs']
        empty_text_blob = [i for i, text_blob in enumerate(text_blob_lists) if text_blob == ""]
        final_url_lists=[url_list[enum] for enum,i in enumerate(text_blob_lists) if enum not in empty_text_blob]
        final_id_lists=[id_lists[enum] for enum,i in enumerate(id_lists) if enum not in empty_text_blob]
        final_text_blob_lists=[text_blob_lists[enum] for enum,i in enumerate(text_blob_lists) if enum not in empty_text_blob]
        if empty_text_blob:
            for i in empty_text_blob:
                try:
                    response = requests.get(url_list[i], timeout=(5, 1))  # (connect_timeout,read_timeout)
                except Exception:
                    continue
                content_type = response.headers.get("Content-Type")
                if content_type is not None and "image" in content_type:
                    image_list.append(id_lists[i])
                elif content_type is not None and "pdf" not in content_type:
                    pdf_list.append(id_lists[i])        
        # If you need to pass the processed data to other tasks, you can set it as XCom
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_file = os.path.join(script_dir, "config.json")
        with open(config_file, "r") as file:
            config = json.load(file)
        dataframe = pd.DataFrame()
        dataframe["links"] = final_url_lists
        dataframe["class"] = [3 for i in final_url_lists]  # any random class
        dataframe["text"]=final_text_blob_lists
        if len(empty_text_blob)!=len(text_blob_lists):
            encoded_data=encoder = Encoder.from_dict(config, dataframe)
            encoded_data = encoder.encoder()
            encoded_data["ids"]=final_id_lists
        encoded_data_json = encoded_data.to_json(orient="split")
        pdf_json = json.dumps(pdf_list)
        image_json = json.dumps(image_list)
        id_json=json.dumps(final_id_lists)
        return {
            "encoded_data": encoded_data_json,
            "pdf_json": pdf_json,
            "image_json": image_json,
            "config": config,
            "id_json":id_json
        }

    @task(max_active_tis_per_dag=1)
    def model_inference(process_conf, **kwargs):
        import pandas as pd
        from model import ModelBert
        from test_predictions import TestPredictor
        from load_dataset import DataLoad
        import boto3
        config = kwargs.get("dag_run").conf
        encoded_data_json = process_conf["encoded_data"]
        pdf_json = process_conf["pdf_json"]
        image_json = process_conf["image_json"]
        id_json=process_conf["id_json"]
        config = process_conf["config"]
        encoded_data = pd.read_json(encoded_data_json, orient="split")
        prediction = {}
        run_config = kwargs.get("dag_run").conf
        encoded_data, pdf_json = process_conf["encoded_data"], process_conf["pdf_json"]
        encoded_data = pd.read_json(encoded_data_json, orient="split")
        image_json, config = process_conf["image_json"], process_conf["config"]
        id_json=process_conf["id_json"]
        image_lists, pdf_lists = json.loads(image_json), json.loads(pdf_json)
        id_lists=json.loads(id_json)
        config = process_conf["config"]
        prediction={}
        if image_lists:
            for i in image_lists:
                prediction[i]=config.get("webapp").get("image")
        if pdf_lists:
            for i in pdf_lists:
                prediction[i]=config.get("webapp").get("Documentation")
        if len(encoded_data) > 0:
            predictor = TestPredictor.from_dict(config)
            input_ids, attention_masks, links = predictor.tokenize_test_data(
                encoded_data
            )
            loader = DataLoad.from_dict(config)
            loader.dataset(input_ids, attention_masks)
            inference_dataloader = loader.dataloader()
            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.realpath(__file__))
            model_file = os.path.join(
                config["model_parameters"]["model_dir"],config["model_parameters"]["saved_model_name"]
            )
            loaded_model=torch.load(model_file)
            category = predictor.predict_test_data(inference_dataloader, loaded_model)
            for enum, each_category in enumerate(category):
                prediction[id_lists[enum]] = config.get("webapp").get(each_category)
        output = {"prediction": prediction}
        return output

        
    @task(max_active_tis_per_dag=100)
    def create_inference_table_(**kwargs):
        # SQL query template
        sql_template = """
        CREATE TABLE IF NOT EXISTS Document_inference (
        id INT NOT NULL,
        document_type INT NOT NULL,
        UNIQUE(id)
        );"""

        # PostgresOperator to execute the query
        task_create_table = PostgresOperator(
            task_id='create_inference_table',
            sql=sql_template,
            # parameters=parameters_list,
            postgres_conn_id='RDS_connection',
            dag=dag,
            autocommit=True
        )
        # Execute the operator
        task_create_table.execute(context=kwargs)

    @task(max_active_tis_per_dag=2)
    def update_inference_table(finale_process,**kwargs):
        prediction=finale_process["prediction"]
        id = list(map(int, prediction.keys()))
        result=list(prediction.values())
        # Construct a list of tuples for parameters
        sql_template=""""""
        for i in range(len(id)):
            # SQL query template
            sql_template = sql_template+f"""
            INSERT INTO Document_inference VALUES ({id[i]}, {result[i]});\n
        """

        # PostgresOperator to execute the query
        task_insert_into_table = PostgresOperator(
            task_id='insert_into_table',
            sql=sql_template,
            # parameters=parameters_list,
            postgres_conn_id='RDS_connection',
            dag=dag,
            autocommit=True
        )
        # Execute the operator
        task_insert_into_table.execute(context=kwargs)
    start >> create_inference_table_()
    process_config = start >> process_urls() 
    finale_process = model_inference(process_config) 
    finale_process >> update_inference_table(finale_process) >> stop

chunk_text_blob_dag = DAG(
    dag_id="sinequa_s3_scheduler",
    catchup=False,
    schedule_interval=timedelta(hours=2),
    start_date=datetime(2023,1,1),
)

get_urls_dag_run = TriggerMultiDagRunOperator(
    task_id="get_text_blobs_run",
    dag=chunk_text_blob_dag,
    trigger_dag_id="sinequa_batch_inference",
    python_callable=chunk_text_blobs,
)

