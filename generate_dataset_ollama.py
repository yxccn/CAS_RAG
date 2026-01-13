import json
import os
from llama_index.core.llama_dataset.rag import LabelledRagDataset
from llama_index.core import Document
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 使用 Document 对象包装每个 abstract_text
    documents = [Document(text=doc['abstract_text'], doc_id=doc['id']) for doc in data]
    return documents

# documents = SimpleDirectoryReader("./data").load_data()
documents = load_json_data("./json/docs.json")
# llm = OpenAI(model="gpt-4o-turbo")
llm = Ollama(model="llama3.1:latest", request_timeout=60.0)
dataset_generator = RagDatasetGenerator.from_documents(
    documents,
    llm=llm,
    num_questions_per_chunk=3,
    show_progress=True,
)
# dataset = dataset_generator.generate_questions_from_nodes()
# dataset.save_json("./json/rag_dataset.json")
# examples = dataset.examples
# for i, example in enumerate(examples):
#     contexts = [n[:100] for n in example.reference_contexts]
#     print(f"{i + 1}. {example.query}")
#     print(f"Ground Truth: {example.reference_answer[:100]}...")

dataset_json = "./json/test-dataset.json"
# if not os.path.exists(dataset_json):
#     dataset = dataset_generator.generate_dataset_from_nodes()
#     examples = dataset.examples
#     dataset.save_json(dataset_json)
# else:
#     dataset = LabelledRagDataset.from_json(dataset_json)
#     examples = dataset.examples
dataset = dataset_generator.generate_dataset_from_nodes()
examples = dataset.examples
dataset.save_json(dataset_json)