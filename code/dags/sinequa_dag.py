from airflow import DAG
from airflow.decorators import task
from uuid import uuid4
from airflow.operators.empty import EmptyOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator

from airflow.utils.dates import days_ago
import os
import json
import torch
import requests
from time import sleep
import numpy as np


doc_md_DAG = """
### <span style="color:blue">Airbus metadata conversion DAG</span> 
#### Airbus metadata conversion DAG:
This DAG converts airbus metadata into STAC. 
"""
with DAG(
    dag_id="sinequa_inference",
    doc_md=doc_md_DAG,
    catchup=False,
    schedule=None,
    start_date=days_ago(0),
    params={"id": int, "url":"", "text_blob": ""},
) as dag:
    start = EmptyOperator(task_id="start", dag=dag)
    stop = EmptyOperator(task_id="stop", dag=dag)

    @task(max_active_tis_per_dag=100)
    def process_urls(**kwargs):
        from preprocessing import Preprocessor
        import pandas as pd
        from encoder import Encoder

        config = kwargs.get("dag_run").conf
        url = config["url"]
        text_blob = config["text_blob"]
        # If you need to pass the processed data to other tasks, you can set it as XCom
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_file = os.path.join(script_dir, "config.json")
        # config_file="/opt/airflow/dags/config.json"
        with open(config_file, "r") as file:
            config = json.load(file)
        encoded_data, pdf_list, image_list = "", [], []
        if text_blob:
            encoder = Encoder.from_dict(config, "text_encoder")
            encoded_data = encoder.text_encoder(text_blob)
            # Store the DataFrame as a JSON object
        else:
            response = requests.get(
                url, timeout=(5, 1)
            )  # (connect_timeout,read_timeout)
            content_type = response.headers.get("Content-Type")
            if content_type is not None and "image" in content_type:
                image_list.append(url)
            elif content_type is not None and "pdf" not in content_type:
                pdf_list.append(url)
        pdf_json = json.dumps(pdf_list)
        image_json = json.dumps(image_list)
        return {
            "encoded_data": encoded_data,
            "pdf_json": pdf_json,
            "image_json": image_json,
            "config": config,
        }

    @task(max_active_tis_per_dag=1)
    def model_inference(process_conf, **kwargs):
        """ """
        import pandas as pd
        from model import ModelBert
        from test_predictions import TestPredictor

        run_config = kwargs.get("dag_run").conf
        url = run_config["url"]
        id=run_config["id"]
        encoded_data, pdf_json = process_conf["encoded_data"], process_conf["pdf_json"]
        image_json, config = process_conf["image_json"], process_conf["config"]
        image_lists, pdf_lists = json.loads(image_json), json.loads(pdf_json)
        config = process_conf["config"]
        if image_lists:
            return {"result":{url: config.get("webapp").get("image")},"id":id}
        elif pdf_lists:
            return {"result":{url: config.get("webapp").get("Documentation")},"id":id}
        else:
            predictor = TestPredictor.from_dict(config)
            input_ids, attention_masks = predictor.tokenize_text_data(encoded_data)
            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.realpath(__file__))
            model_file = os.path.join(
                config["model_parameters"]["model_dir"],
                config["model_parameters"]["saved_model_name"],
            )
            #     # model=ModelBert.from_dict(config)
            #     # loaded_model = model.load_model(model_file)
            loaded_model = torch.load(model_file)
            outputs = loaded_model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits
            logits = torch.sigmoid(logits)
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            pred_position = np.argmax(logits).tolist()
            category = predictor.convert_labels_to_class(pred_position)
            prediction = config.get("webapp").get(category)
            return {"result":{url:prediction},"id":id}
        
        # # Your task to insert data into RDS
    create_inference_table = PostgresOperator(
            task_id="create_inference_table",
            sql="sql/create_inference_table.sql",
            postgres_conn_id='RDS_connection'
        )
    
    @task(max_active_tis_per_dag=1)
    def update_inference_table(finale_process,**kwargs):
        prediction=finale_process["result"]
        id=finale_process["id"]
        task_insert_into_table = PostgresOperator(
        task_id='insert_into_table',
        sql=f"""
        INSERT INTO Document_inference VALUES (%s,%s);
        """,
        parameters=[id,list(prediction.values())[0]]
,
        postgres_conn_id='RDS_connection',
        dag=dag,autocommit=True
    )
        task_insert_into_table.execute(context=kwargs)

    process_config = start >> process_urls()
    start >> create_inference_table
    finale_process = model_inference(process_config)
    finale_process >> update_inference_table(finale_process) >> stop