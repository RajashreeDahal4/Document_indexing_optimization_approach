import json

from load_dataset import DataLoad
from test_predictions import TestPredictor


def batch_predicts(config_file, urls):
    """
    Predicts category of each url given a list of urls
    Arg(s):
            config_file: json file for config
            urls (list): The URL of the test data in the form of list
    Returns:
        tuple: Tuple of dictionary with key as url and its value as class category, and lists of url with pdf url_type
    """
    with open(config_file, "rb") as files:
        config = json.load(files)
    predictor = TestPredictor.from_dict(config)
    prediction = {}
    encoded_data, pdf_lists, image_lists = predictor.process_test_data(urls)
    if len(encoded_data) > 0:
        input_ids, attention_masks, links = predictor.tokenize_test_data(encoded_data)
        loader = DataLoad.from_dict(config)
        loader.dataset(input_ids, attention_masks)
        inference_dataloader = loader.dataloader()
        category = predictor.predict_test_data(inference_dataloader)
        for enum, each_category in enumerate(category):
            prediction[links[enum]] = config.get("webapp").get(each_category)
    for image_url in image_lists:
        prediction[image_url] = config.get("webapp").get("image")
    for pdf_url in pdf_lists:
        prediction[pdf_url] = config.get("webapp").get("Documentation")
    return prediction, pdf_lists


# result=batch_predicts("code/config.json",['https://www.facebook.com'])
