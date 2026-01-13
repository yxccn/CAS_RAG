import re
import time
import subprocess
from tqdm import tqdm
from llama_index.core import Document
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llama_dataset.rag import LabelledRagDataset
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts import PromptTemplate
from llama_index.core.evaluation.answer_relevancy import DEFAULT_EVAL_TEMPLATE
from llama_index.core.evaluation import AnswerRelevancyEvaluator, RelevancyEvaluator, CorrectnessEvaluator, FaithfulnessEvaluator
from llama_index.core.evaluation import BatchEvalRunner

eval_template = DEFAULT_EVAL_TEMPLATE
llm = Ollama(model="llama3.1:latest", request_timeout=60.0)

dataset_json = "./json/test-dataset.json"
dataset = LabelledRagDataset.from_json(dataset_json)
examples = dataset.examples
total_score = 0
num_examples = len(examples)
##################################################RAG
start_time = time.time()
with tqdm(total=num_examples, desc="Evaluating RAG") as pbar:
    for i, example in enumerate(examples):
        question = example.query
        result_score = None
        while result_score is None:
            result = subprocess.run(
                ['/home/yxc/anaconda3/envs/ragevi/bin/python', 'query.py', f'--question="{question}"'],
                capture_output=True, text=True
            )
            # print(result)
            # 获取输出并提取 'query_answer:' 后的内容
            output = result.stdout.strip()
            if "query_answer:" in output:
                answer = output.split("query_answer:")[1].strip()
            else:
                answer = "No answer found"

            ##################################################eval
            evaluator = AnswerRelevancyEvaluator(llm, eval_template=eval_template)
            result = evaluator.evaluate(query=question, response=answer)
            result_score = result.score
            if result_score is None:
                print(f"Score for question '{question}' was None, re-evaluating...")
        total_score += result.score
        # Update the progress bar
        pbar.update(1)
        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / (i + 1)) * num_examples
        pbar.set_postfix(remaining_time=f"{estimated_total_time - elapsed_time:.2f}s")

average_score = total_score / num_examples
print(f"\nAnswer Relevancy Average Score: {average_score}")