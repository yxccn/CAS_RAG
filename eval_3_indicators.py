import re
import time
from tqdm import tqdm
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.llama_dataset.rag import LabelledRagDataset
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.evaluation import RelevancyEvaluator, CorrectnessEvaluator, FaithfulnessEvaluator
from llama_index.core.evaluation import BatchEvalRunner
from transformers import AutoModel
from llama_index.embeddings.ollama import OllamaEmbedding

# 加载模型
# local_embedding_model = HuggingFaceEmbedding(model_name="./models/e5-large", device="cuda:2")
# local_embedding_model = HuggingFaceEmbedding(model_name="./models/UAE-Large-V1", device="cuda:2")
# local_embedding_model = OllamaEmbedding(model_name="nomic-embed-text")
# local_embedding_model = OllamaEmbedding(model_name="mxbai-embed-large")
# local_embedding_model = HuggingFaceEmbedding(model_name="./models/bge-large-zh-v1.5", device="cuda:2")
local_embedding_model = HuggingFaceEmbedding(model_name="./models/gte-Qwen2-1.5B-instruct", device="cuda:2")    # default
# local_embedding_model = HuggingFaceEmbedding(model_name="./models/gte-Qwen2-7B-instruct", device="cuda:2")
# local_embedding_model = HuggingFaceEmbedding(model_name="./models/Conan-embedding-v1", device="cuda:2")
# local_embedding_model = HuggingFaceEmbedding(model_name="./models/all-MiniLM-L12-v2", device="cuda:2")

# llm = Ollama(model="qwen2.5:3b", request_timeout=600.0)
# llm = Ollama(model="qwen2.5:7b", request_timeout=600.0)
# llm = Ollama(model="qwen2.5:72b", request_timeout=600.0)
# llm = Ollama(model="llama3.2:3b", request_timeout=600.0)
llm = Ollama(model="llama3.1:8b", request_timeout=600.0)    # default
# llm = Ollama(model="gemma2:2b", request_timeout=600.0)
# llm = Ollama(model="gemma2:9b", request_timeout=600.0)
# llm = Ollama(model="gemma2:27b", request_timeout=600.0)
# llm = Ollama(model="mistral:7b", request_timeout=600.0)

#定义评估器
relevancy_evaluator = RelevancyEvaluator(llm)
correctness_evaluator = CorrectnessEvaluator(llm)
faithfulness_evaluator = FaithfulnessEvaluator(llm)
# 批量评估设置
runner = BatchEvalRunner(
    evaluators={
        "relevancy": relevancy_evaluator,
        "correctness": correctness_evaluator,
        "faithfulness": faithfulness_evaluator,
    },
    workers=8,
)

#定义数据
dataset_json = "./json/test-dataset.json"
dataset = LabelledRagDataset.from_json(dataset_json)
examples = dataset.examples
num_examples = len(examples)
#初始化
total_score = {"relevancy": 0.0, "correctness": 0.0, "faithfulness": 0.0}
average_scores = {"relevancy": 0.0, "correctness": 0.0, "faithfulness": 0.0}
questions = [example.query for example in examples]
ground_truths = [example.reference_answer for example in examples]

# 初始化RAG
similarity_top_k = 3    # 1-15,default 3
PERSIST_DIR = "./storage"
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context, embed_model=local_embedding_model)
full_query_engine = index.as_query_engine(similarity_top_k=similarity_top_k, llm=llm)
prompt = (
    "Answer this civil aviation practitioners's question in JSON format, with the keys 'answer' and 'references'. "
    "Provide the answer in a string under 'answer' and references following the AMA format under 'references'.\n"
)

start_time = time.time()
with tqdm(total=num_examples, desc="Evaluating RAG") as pbar:
    for i, example in enumerate(examples):
        question = example.query
        ground_truth = example.reference_answer

        ###eval
        metrics_results = runner.evaluate_queries(
            full_query_engine, queries=[question], reference=[ground_truth]
        )
        for metric in metrics_results.keys():
            eval_result = metrics_results[metric][0]
            total_score[metric] += eval_result.score
            average_scores[metric] = total_score[metric] / (i + 1)
        #更新pbar
        pbar.set_postfix({
            "Relevancy Avg": f"{average_scores['relevancy']:.2f}",
            "Correctness Avg": f"{average_scores['correctness']/5:.2f}",
            "Faithfulness Avg": f"{average_scores['faithfulness']:.2f}"
        })
        pbar.update(1)

average_scores["correctness"] /= 5
for metric, avg_score in average_scores.items():
    print(f"\n{metric} Average Score: {avg_score}")