from airflow import DAG
from airflow.decorators import task
from airflow_multi_dagrun.operators import TriggerMultiDagRunOperator
from uuid import uuid4
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
from dateutil.parser import parse
import os
import json
import torch

from airflow.hooks.base_hook import BaseHook

from time import sleep

doc_md_DAG = """
### <span style="color:blue">Airbus metadata conversion DAG</span> 
#### Airbus metadata conversion DAG:
This DAG converts airbus metadata into STAC. 
"""


def get_urls_to_process(**kwargs):
    """ """
    config = kwargs["dag_run"].conf
    urls = config["urls"]
    collection_id = config["collection_id"]
    chunk_size = config.get("chunk_size", 12)
    url_chunks = [urls[i : i + chunk_size] for i in range(0, len(urls), chunk_size)]
    for chunks in url_chunks:
        yield {"urls": chunks, "collection_id": collection_id}
        sleep(1)


with DAG(
    dag_id="url_processor",
    doc_md=doc_md_DAG,
    catchup=False,
    schedule=None,
    start_date=days_ago(0),
    params={"urls": [], "chunk_size": 12, "collection_id": int()},
) as dag:
    start = EmptyOperator(task_id="start", dag=dag)
    stop = EmptyOperator(task_id="stop", dag=dag)

    @task(max_active_tis_per_dag=100)
    def process_urls(**kwargs):
        """ """
        from preprocessing import Preprocessor
        import pandas as pd
        from encoder import Encoder

        config = kwargs.get("dag_run").conf
        url_list = config["urls"]
        # If you need to pass the processed data to other tasks, you can set it as XCom
        script_dir = os.path.dirname(os.path.realpath(__file__))
        print("THe script directory is", script_dir)
        config_file = os.path.join(script_dir, "config.json")
        with open(config_file, "r") as file:
            config = json.load(file)
        dataframe = pd.DataFrame()
        dataframe["links"] = url_list
        dataframe["class"] = [3 for i in url_list]  # any random class
        processor = Preprocessor.from_dict(config, dataframe)
        (dataframe, pdf_lists, image_lists) = processor.preprocessed_features()
        dataframe["text"] = dataframe["soup"]
        encoder = Encoder.from_dict(config, dataframe)
        encoded_data = encoder.encoder()
        # Store the DataFrame as a JSON object
        encoded_data_json = encoded_data.to_json(orient="split")

        pdf_json = json.dumps(pdf_lists)
        image_json = json.dumps(image_lists)

        return {
            "encoded_data_json": encoded_data_json,
            "pdf_json": pdf_json,
            "image_json": image_json,
            "config": config,
        }

    @task(max_active_tis_per_dag=1)
    def model_inference(process_conf, **kwargs):
        """ """
        import boto3
        import pandas as pd
        from model import ModelBert
        from test_predictions import TestPredictor
        from load_dataset import DataLoad

        config = kwargs.get("dag_run").conf
        collection_id = config["collection_id"]
        encoded_data_json = process_conf["encoded_data_json"]
        pdf_json = process_conf["pdf_json"]
        image_json = process_conf["image_json"]
        config = process_conf["config"]
        encoded_data = pd.read_json(encoded_data_json, orient="split")
        prediction = {}
        image_lists, pdf_lists = [], []
        if len(encoded_data) > 0:
            predictor = TestPredictor.from_dict(config)
            input_ids, attention_masks, links = predictor.tokenize_test_data(
                encoded_data
            )
            loader = DataLoad.from_dict(config)
            loader.dataset(input_ids, attention_masks)
            inference_dataloader = loader.dataloader()
            pdf_lists = json.loads(pdf_json)
            image_lists = json.loads(image_json)
            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.realpath(__file__))
            model_file = os.path.join(
                config["model_parameters"]["model_dir"], config["model_parameters"]["saved_model_name"]
            )
            print("I am going to load the model================")
            loaded_model=torch.load(model_file)
            category = predictor.predict_test_data(inference_dataloader, loaded_model)
            for enum, each_category in enumerate(category):
                prediction[links[enum]] = config.get("webapp").get(each_category)
        for image_url in image_lists:
            prediction[image_url] = config.get("webapp").get("image")
        for pdf_url in pdf_lists:
            prediction[pdf_url] = config.get("webapp").get("Documentation")
        output = {
            "prediction": prediction,
            "pdf_lists": pdf_lists,
            "collection_id": collection_id,
        }
        print("The result is", output)
        return output

    process_config = start >> process_urls()
    finale_process = model_inference(process_config)

    finale_process >> stop


get_urls_dag = DAG(
    dag_id="chunk_urls_to_process",
    max_active_runs=4,
    start_date=days_ago(1),
    schedule=None,
    params={"urls": [], "chunk_size": 12, "collection_id": int()},
)

get_urls_dag_run = TriggerMultiDagRunOperator(
    task_id="get_urls_dag_run",
    dag=get_urls_dag,
    trigger_dag_id="url_processor",
    python_callable=get_urls_to_process,
)
