import pickle

# 打开pkl文件并加载数据
with open('./pickle/pubmed_abstract_sample.pkl', 'rb') as file:
    data = pickle.load(file)

# 打印加载的数据
print(data)
