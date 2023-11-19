import os
import torch.nn as nn
import json
import inspect
from model import ModelBert
from test_predictions import TestPredictor
from load_dataset import DataLoad

def model_fn(model_dir):
    print("Rajashree is inside model_fn")
    current_file_path = inspect.getfile(lambda: None)
    config_file_path='/opt/ml/model/code/config.json'
    with open(config_file_path, "rb") as files:
        config = json.load(files)
    loaded_model=ModelBert.from_dict(config)
    model = loaded_model.load_model(model_dir)
    print("Rajashree model loadded successful")
    return model

def predict_fn(input_data, model):
    prediction = {}
    print("I am in predict_fn")
    encoded_data, pdf_lists, image_lists,predictor,config=input_data
    if len(encoded_data) > 0:
        input_ids, attention_masks, links = predictor.tokenize_test_data(encoded_data)
        loader = DataLoad.from_dict(config)
        loader.dataset(input_ids, attention_masks)
        inference_dataloader = loader.dataloader()
        category = predictor.predict_test_data(inference_dataloader,model)
        for enum, each_category in enumerate(category):
            prediction[links[enum]] = config.get("webapp").get(each_category)
    for image_url in image_lists:
        prediction[image_url] = config.get("webapp").get("image")
    for pdf_url in pdf_lists:
        prediction[pdf_url] = config.get("webapp").get("Documentation")
    outputs=(prediction,pdf_lists)
    return  {"out":outputs}
    

def output_fn(prediction, content_type):
    if content_type == 'application/json':
        return json.dumps(prediction)
    raise Exception('Unsupported Content Type')

def input_fn(request_body, request_content_type):
    print("I am in input_fn")
    print("The request_body is",request_body)
    print("type of request body",type(request_body))
    print("the request content type is",request_content_type)
    if request_content_type == 'application/json':
        urls_dict = json.loads(request_body)
        urls=urls_dict['urls']
        print("The urls list is",urls)
        config_file_path='/opt/ml/model/code/config.json'
        with open(config_file_path, "rb") as files:
            config = json.load(files)
        print("Config file loaded")
        print("config file is",config)
        print(type(config))
        predictor = TestPredictor.from_dict(config)
        print("predictor assigned")
        prediction = {}
        encoded_data, pdf_lists, image_lists = predictor.process_test_data(urls)
        print("Process part done")
        input_data=(encoded_data,pdf_lists,image_lists,predictor,config)
        return input_data
    else:
        raise Exception('Unsupported Content Type')