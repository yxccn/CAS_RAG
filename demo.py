import re
import time
from tqdm import tqdm
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.llama_dataset.rag import LabelledRagDataset
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.evaluation import RelevancyEvaluator, CorrectnessEvaluator, FaithfulnessEvaluator
from llama_index.core.evaluation import BatchEvalRunner
from langchain.embeddings import HuggingFaceEmbeddings

# 加载模型
local_embedding_model = HuggingFaceEmbedding(model_name="/home/yxc/rag/models/gte-Qwen2-1.5B-instruct")
llm = Ollama(model="llama3.1:latest", request_timeout=60.0)

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

start_time = time.time()
with tqdm(total=num_examples, desc="Evaluating RAG") as pbar:
    for i, example in enumerate(examples):
        question = example.query
        ground_truth = example.reference_answer

        ##########################################################RAG
        similarity_top_k = 3
        PERSIST_DIR = "./storage"

        def extract_response_from_exception_message(exception_message):
            match = re.search(r'Response:\s*(.*)', exception_message, re.DOTALL)
            if match:
                return match.group(1).strip()
            else:
                return "No valid response found."

        def query_llama(prompt, question):
            try:
                response = full_query_engine.query(prompt + question)
            except ValueError as e:
                if 'did not respond with valid JSON' in str(e):
                    response = extract_response_from_exception_message(str(e))
                else:
                    raise e

            return response

        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context, embed_model=local_embedding_model)

        full_query_engine = index.as_query_engine(similarity_top_k=similarity_top_k, llm=llm)
        prompt = (
            "Answer this civil aviation practitioners's question in JSON format, with the keys 'answer' and 'references'. "
            "Provide the answer in a string under 'answer' and references following the AMA format under 'references'.\n"
        )
        response = query_llama(
            "Answer this civil aviation practitioners's question and provide references at the end of your responses. The references should follow the AMA format: \n",
            question
        )
        answer = str(response)
        ##################################################eval
        metrics_results = runner.evaluate_queries(
            full_query_engine, queries=questions, reference=ground_truths
        )
        for metric in metrics_results.keys():
            eval_result = metrics_results[metric][0]
            total_score[metric] += eval_result.score
            average_scores[metric] = total_score[metric] / (i + 1)
        #更新pbar
        pbar.set_postfix({
            "Relevancy Avg": f"{average_scores['relevancy']:.2f}",
            "Correctness Avg": f"{average_scores['correctness']:.2f}",
            "Faithfulness Avg": f"{average_scores['faithfulness']:.2f}"
        })
        pbar.update(1)

average_scores["correctness"] /= 5
for metric, avg_score in average_scores.items():
    print(f"\n{metric} Average Score: {avg_score}")