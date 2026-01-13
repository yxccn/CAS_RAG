import pickle
import json

# 步骤一：加载 .pkl 文件
with open('./pickle/pubmed_abstract_sample.pkl', 'rb') as pkl_file:
    data = pickle.load(pkl_file)  # 将pkl文件中的内容加载为Python对象

# 步骤二：将加载的Python对象保存为 .json 文件
with open('./json/pubmed_abstract_sample.json', 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)  # indent参数使JSON格式更易读
