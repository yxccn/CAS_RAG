import json

# 加载JSON文件
file_path = 'json/civil_aviation_safety_abstract.json'
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 统计顶级键（项）的数量
num_items = len(data)
print(num_items)
