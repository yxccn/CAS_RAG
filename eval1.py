import json
import subprocess
from llama_index.core import Document
from llama_index.core.evaluation import RelevancyEvaluator, CorrectnessEvaluator, FaithfulnessEvaluator
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llama_dataset.rag import LabelledRagDataset
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts import PromptTemplate
from llama_index.core.evaluation.answer_relevancy import DEFAULT_EVAL_TEMPLATE
from langchain.embeddings import HuggingFaceEmbeddings

translate_prompt = "\n\nPlease reply in Chinese."
eval_template = DEFAULT_EVAL_TEMPLATE
eval_template.template += translate_prompt
# eval_template = DEFAULT_EVAL_TEMPLATE + "\n\nPlease reply in Chinese."

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 使用 Document 对象包装每个 abstract_text
    documents = [Document(text=doc['abstract_text'], doc_id=doc['id']) for doc in data]
    return documents

# local_embedding_model = HuggingFaceEmbedding(model_name="/home/yxc/rag/models/gte-Qwen2-1.5B-instruct")

llm = Ollama(model="llama3.1:latest", request_timeout=60.0)
documents = load_json_data("./json/docs.json")

dataset_json = "./json/test-dataset.json"
dataset = LabelledRagDataset.from_json(dataset_json)
examples = dataset.examples
question = examples[2].query


local_embedding_model = HuggingFaceEmbeddings(
    model_name="/home/yxc/rag/models/gte-Qwen2-1.5B-instruct",
    model_kwargs={'device': 2},
    encode_kwargs={'normalize_embeddings': True}
)
##################################################RAG
# node_parser = SentenceSplitter()
# nodes = node_parser.get_nodes_from_documents(documents)
# Settings.llm = llm
# vector_index = VectorStoreIndex(nodes, embed_model=local_embedding_model)
# engine = vector_index.as_query_engine()
# response = engine.query(question)
# answer = str(response)
############################################################add rag
import re
from llama_index.core import Document, StorageContext, load_index_from_storage, SimpleDirectoryReader, VectorStoreIndex

similarity_top_k = 3
PERSIST_DIR = "./storage"

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

storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context, embed_model=local_embedding_model)

full_query_engine = index.as_query_engine(similarity_top_k=similarity_top_k, llm=llm)
prompt = (
    "Answer this patient's question in JSON format, with the keys 'answer' and 'references'. "
    "Provide the answer in a string under 'answer' and references following the AMA format under 'references'.\n"
)
response = query_llama(
    "Answer this patient's question and provide references at the end of your responses. The references should follow the AMA format: \n",
    question
)
answer = str(response)
##################################################eval
# print(f"Question: {question}")
# print(f"Answer: {answer}")
# evaluator = ContextRelevancyEvaluator(llm)
# # result = evaluator.evaluate(query=question, response=answer)
# contexts = [n.get_content() for n in response.source_nodes]
# result = evaluator.evaluate(query=question, contexts=contexts)
# print(f"Score: {result.score}")
# print(f"Feedback: {result.feedback}")
# # print(f"Contexts: {result.contexts}")
# print(f"\n\nresult: {result}")

from llama_index.core.evaluation import BatchEvalRunner

# answer_relevancy_evaluator = AnswerRelevancyEvaluator(llm)
# context_relevancy_evaluator = ContextRelevancyEvaluator(llm)
relevant_evaluator = RelevancyEvaluator(llm)
correctness_evaluator = CorrectnessEvaluator(llm)
faithfulness_evaluator = FaithfulnessEvaluator(llm)

runner = BatchEvalRunner(
    evaluators={
        "relevancy": relevant_evaluator,
        "correctness": correctness_evaluator,
        "faithfulness": faithfulness_evaluator,
    },
    workers=8,
)
questions = [example.query for example in examples]
ground_truths = [example.reference_answer for example in examples]
metrics_results = runner.evaluate_queries(
    full_query_engine, queries=questions, reference=ground_truths
)

for metrics in metrics_results.keys():
    print(f"metrics: {metrics}")
    eval_results = metrics_results[metrics]
    for eval_result in eval_results:
        print(f"score: {eval_result.score}")
        print(f"feedback: {eval_result.feedback}")
        if eval_result.passing is not None:
            print(f"passing: {eval_result.passing}")