"""This file is used to run object detection and object classification
from images.

You can run this script from console or in a code editor.

Example
-------
python pipeline_json.py

Note
-------
Output files will be stored in folder data/processed/

"""

import os

import sys;sys.path.append('..')
from utils.abs_path import abs_path

# IMPORT MODULES WITH MODELS
import inference.object_detection as od
import inference.object_classification as cl
import csv
import json
import pandas as pd
from transliterate import slugify

# MODEL'S PARAMETERS

# 1. OBJECT DETECTION
od.DEFAULT_OD_MIN_SCORE = 0.01
od.DEFAULT_MODEL_PATH = abs_path('../../models/OD/henkel')
od.DEFAULT_MODEL_CFG_FILE = od.DEFAULT_MODEL_PATH + '/yolo-obj.cfg'
od.DEFAULT_MODEL_WEIGHT_FILE = od.DEFAULT_MODEL_PATH + '/yolo-obj.weights'
od.DEFAULT_MODEL_CLASSES_FILE = od.DEFAULT_MODEL_PATH + '/obj.names'
od.DEFAULT_OUTPUT_IMAGE_WITH_BOXES_FOLDER = abs_path('../../data/processed/img_od_boxes')
od.DEFAULT_OUTPUT_JSON_WITH_BOXES_FOLDER = abs_path('../../data/processed/json_od_boxes')

# 2. Classification
cl.DEFAULT_CL_MAX_DISTANCE = 0.5
cl.DEFAULT_MODEL_PATH = abs_path('../../models/CL/resnet152-b121ed2d.pth')
cl.DEFAULT_CLUSTER_DATA_FILE = abs_path('../../models/CL/henkel/Henkel_clusters.data')
cl.DEFAULT_CLASSES_LIST = abs_path('../../models/CL/henkel/map_class_to_id.json')
cl.DEFAULT_OUTPUT_IMAGE_CLASS_CUT_FOLDER = abs_path('../../data/processed/img_cl_cutted')
cl.DEFAULT_OUTPUT_JSON_FOLDER = abs_path('../../data/processed/json_output')
cl.DEFAULT_MODEL_PICK_LAYER = 'avg'  # extract feature of this layer
cl.DEFAULT_DISTANCE_TYPE = 'd1'
cl.DEFAULS_RETRIEVED_DEPTH = 1

OUTPUT_FOLDER_EVAL = abs_path('../../data/eval_output')


def take_image_from_json(json_file):
    """return list of images, save file with hand-boxing
    :rtype: list
    """
    with open(json_file, 'rb', encoding=None) as f:
        d = json.load(f)

    parsed_json = pd.io.json.json_normalize(d,
                                            record_path=['marks', 'mark_coordinate'], record_prefix='coordinates_',
                                            meta=['id', 'created_at', 'photo', ['marks', 'id'],
                                                  ['marks', 'attribute_type']]
                                            , errors='ignore'
                                            )

    parsed_marks = pd.io.json.json_normalize(d,
                                             record_path=['marks'],
                                             errors='ignore'
                                             )

    files = []  # getting links and name of files
    labels = []

    if not parsed_marks.empty:
        parsed_marks.drop(['attribute_type', 'mark_coordinate', 'active'], axis=1, inplace=True)
        parsed_marks.rename(columns={"id": "marks_id"}, inplace=True)

        processed_json = pd.concat([parsed_json, parsed_marks], axis=1, sort=False)
        processed_json = processed_json[processed_json['marks.attribute_type'] == 'sku']
        processed_json['local_link'] = os.getcwd() + '/' + processed_json["photo"].str.slice(7)

        for i in enumerate(processed_json.id.unique()):

            one_image_data = processed_json[processed_json.id == i[1]]

            # save image from web
            pil_image_link = one_image_data.iloc[0]['photo']
            orig_img_name = one_image_data.iloc[0]['local_link']
            orig_img_name = orig_img_name.split('/')[-1]

            files.append(pil_image_link)

            for idx, val in enumerate(one_image_data.index):
                parsed_marks = one_image_data[one_image_data.index == val]
                w = (parsed_marks.iloc[0]['coordinates_left'] + parsed_marks.iloc[0]['coordinates_width'])
                h = (parsed_marks.iloc[0]['coordinates_top'] + parsed_marks.iloc[0]['coordinates_height'])
                x = parsed_marks.iloc[0]['coordinates_left']
                y = parsed_marks.iloc[0]['coordinates_top']

                label = ["hand", 0, x, y, w, h, orig_img_name, slugify(parsed_marks.iloc[0]['product.name'])]
                labels.append(label)

    return files, labels


def main(json_path):
    """Processes a image (one or folder of images) and produce JSON file
        with classified objects.

    Args:
        image_path (str, default None):
            Processes a image (one or folder of images).

    Returns:
        JSON file with detected class ids and coordinates of objects. 
        For example, [{"object": "a02ace99-571c-4bb6-80cd-38869c6f402a", 
        "coordinate": {"x1": 0.115, "x2": 0.251875, "y1": 0.19866385372714487, "y2": 0.39451476793248946}}].

    """

    object_detector = od.ObjectDetector()
    object_classification = cl.ObjectClassification()

    json_files = []
    for r, d, f in os.walk(json_path):
        for file in f:
            if not 'checkpoint' in file:
                json_files.append(os.path.join(r, file))

    print("JSON files count: ", len(json_files))

    eval_file = open(OUTPUT_FOLDER_EVAL + '/' + 'eval.csv', mode='w', newline='\n')
    writer_file = csv.writer(eval_file)

    for json_file in json_files:

        # TAKE HAND MARKS
        files, labels = take_image_from_json(json_file=json_file)

        print(json_file)

        # DETECT OBJECTS
        for file in files:
            pred_boxes, width, height = object_detector.predict(file)

            orig_img_name = file.split('/')[-1]
            _ = [row.append(width) for row in labels if row[6] == orig_img_name]
            _ = [row.append(height) for row in labels if row[6] == orig_img_name]

            # OBJECT CLASSIFICATIONN
            if pred_boxes:
                if pred_boxes[0]:
                    output = object_classification.predict(file, pred_boxes)
                    writer_file.writerows(output)

        if labels:
            writer_file.writerows(labels)

    eval_file.close()

if __name__ == '__main__':
    import argparse

    # PARAMETERS
    json_folder_path = abs_path('../../data/input/json')

    main(json_path = json_folder_path)
