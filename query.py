##################################### Import libraries #####################################
import os
import argparse
from llama_index.core import Document, StorageContext, load_index_from_storage, SimpleDirectoryReader, VectorStoreIndex
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_experimental.llms.ollama_functions import OllamaFunctions

##################################### Parse arguments #####################################

parser = argparse.ArgumentParser(description='Query with custom parameters.')
parser.add_argument('--similarity_top_k', type=int, default=3, help='Number of selected references')
parser.add_argument('--question', type=str, required=True, help='The question to query')
args = parser.parse_args()

similarity_top_k = args.similarity_top_k
question = args.question

##################################### Set up path, key and var #####################################

PERSIST_DIR = "./storage"
# similarity_top_k = 10  # Number of selected references

##################################### functions #####################################
import re

def extract_response_from_exception_message(exception_message):
    # 使用正则表达式提取 "Response:" 之后的所有内容
    match = re.search(r'Response:\s*(.*)', exception_message, re.DOTALL)
    if match:
        return match.group(1).strip()  # 获取匹配到的内容并去掉首尾空白符
    else:
        return "No valid response found."  # 如果没有找到 "Response:"，返回默认消息

# def query_llama(prompt, question):
#     response = full_query_engine.query(prompt + question)
#     return response
def query_llama(prompt, question):
    try:
        # 查询 LLM，得到完整的响应
        response = full_query_engine.query(prompt + question)
    except ValueError as e:
        # 当 JSON 解析失败时，提取原始内容
        # 检查错误消息是否包含特定提示，以确保正确捕获相关错误
        if 'did not respond with valid JSON' in str(e):
            # 从错误消息中提取 Response: 后面的部分
            response = extract_response_from_exception_message(str(e))
        else:
            raise e  # 如果是其他类型的错误，继续抛出

    return response


##################################### Load index #####################################
local_embedding_model = HuggingFaceEmbeddings(
    model_name="/home/yxc/rag/models/gte-Qwen2-1.5B-instruct",
    model_kwargs={'device': 2},
    encode_kwargs={'normalize_embeddings': True}
)
# load the existing index
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context, embed_model=local_embedding_model)

##################################### Query #####################################
# 使用 Ollama 加载 llama3.1 模型
local_llm = OllamaFunctions(model="llama3.1:latest")

# 配置 query 引擎
full_query_engine = index.as_query_engine(similarity_top_k=similarity_top_k, llm=local_llm)

prompt = (
    "Answer this patient's question in JSON format, with the keys 'answer' and 'references'. "
    "Provide the answer in a string under 'answer' and references following the AMA format under 'references'.\n"
)

# query an example question
query_answer = query_llama(
    "Answer this patient's question and provide references at the end of your responses. The references should follow the AMA format: \n",
    question
)

print("query_answer:", query_answer)