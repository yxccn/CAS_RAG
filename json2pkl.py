import json
import pickle

# 步骤一：从 .json 文件读取数据
with open('./json/civil_aviation_safety_abstract.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)  # 将 JSON 文件中的内容加载为 Python 对象

# 步骤二：将 Python 对象保存为 .pkl 文件
with open('./pkl/civil_aviation_safety_abstract.pkl', 'wb') as pkl_file:
    pickle.dump(data, pkl_file)  # 将 Python 对象序列化为 .pkl 文件
