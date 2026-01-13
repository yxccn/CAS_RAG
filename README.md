# title

## Install dependencies
```sh
conda create -n casrag python==3.11.0
pip install --no-deps -r requirements.txt
conda activate casrag
```

## Prepare data
Place all data as json file into the `./json` folder. Replace all the directions in `json2pkl.py`, then run`python json2pkl.py` to convert json files into pkl files. Those pkl files will be used to build index.

## Build index
Run `build_index.py` to build index from documents and save it in `./storage`.

## Query
Run `python query.py --similarity_top_k=3 --question="your query"` to retrieve answers with the top 3 selected references.

## Evaluation
Run `python eval_3_indicators.py` to evaluate the performance of the model.

## Model sources
### LLMs
```sh
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
ollama pull qwen2.5:72b
ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull gemma2:2b
ollama pull gemma2:9b
ollama pull gemma2:27b
ollama pull mistral:7b
```

### Embedding models
```sh
huggingface-cli download --resume-download intfloat/e5-large --local-dir ./models/e5-large
huggingface-cli download --resume-download WhereIsAI/UAE-Large-V1 --local-dir ./models/UAE-Large-V1
ollama pull nomic-embed-text
ollama pull mxbai-embed-large
huggingface-cli download --resume-download BAAI/bge-large-zh-v1.5 --local-dir ./models/bge-large-zh-v1.5
huggingface-cli download --resume-download Alibaba-NLP/gte-Qwen2-1.5B-instruct --local-dir ./models/gte-Qwen2-1.5B-instruct
huggingface-cli download --resume-download Alibaba-NLP/gte-Qwen2-7B-instruct --local-dir ./models/gte-Qwen2-7B-instruct
huggingface-cli download --resume-download TencentBAC/Conan-embedding-v1 --local-dir ./models/Conan-embedding-v1
huggingface-cli download --resume-download sentence-transformers/all-MiniLM-L12-v2 --local-dir ./models/all-MiniLM-L12-v2
```