# json2csv.py는 label이 각각 json파일로 구성되었을 때 모든 label을 하나의 csv 파일로 만드는 코드입니다.

import os
import pandas as pd
import json

# directory에 있는 파일들의 이름을 불러와 리스트에 저장하는 함수
def get_file_list(directory):
    file_list = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            file_list.append(filename)
    return file_list


def json2csv(label_path, output_path):
    label_datas = get_file_list(label_path)

    for label_data in label_datas:
        print(label_data)


# jsonfile 읽는 함수
def json2csv(label_path, output_path):

    # 폴더 내의 모든 파일 리스트를 얻기
    files = os.listdir(label_path)

    # JSON 파일만 필터링
    json_files = [f for f in files if f.endswith('.json')]

    data_dict = {'File_name': [], 'Class': []}
    # 각 JSON 파일에 대해 반복
    for json_file in json_files:
        file_path = os.path.join(label_path, json_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            # JSON 파일 읽기
            data = json.load(f)

            # 여기에서 data를 사용하여 원하는 작업 수행
            # print(f"Contents of {json_file}:")
            # print(data)
            # print("\n")

            source = data['SourceDataInfo']['SourceDataId']  # customize
            label = data['LabelDataInfo']['Class'] # customize
            data_dict['File_name'].append(source) 
            data_dict['Class'].append(label)

    df = pd.DataFrame(data_dict)
    df.to_csv(output_path, index=False)
    print("< label csv file uploaded >")


label_path = '/home/elicer/Exercise/label'   # customize
output_path = '/home/elicer/Exercise/csv/label.csv'   # customize
json2csv(label_path, output_path)