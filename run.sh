#!/bin/bash

# model relative
llm_path="models/ggml-model-q8_0.gguf"
n_gpu_layers=-1
n_ctx=10000 
max_tokens=10000
temperature=0.0
embedding_path="models/gte-large-zh"
reranker_path="models/bge-reranker-base"
asr_model_path="models/faster-whisper-small"
tts_model_path="models/full_band_vits"

# retreival relative
docs_path="data/wiki"
db_path="faiss_db/wiki"
stage1_top_k=20
stage2_top_k=3

# run relative ['chat', 'agent']
mode=chat
agent_max_iters=5
debug=false

python main_asr.py \
    --mode "$mode" \
    --llm_path "$llm_path" \
    --n_gpu_layers "$n_gpu_layers" \
    --n_ctx "$n_ctx" \
    --asr_model_path "$asr_model_path" \
    --tts_model_path "$tts_model_path"
    --max_tokens "$max_tokens" \
    --temperature "$temperature" \
    --embedding_path "$embedding_path" \
    --reranker_path "$reranker_path" \
    --db_path "$db_path" \
    --docs_path "$docs_path" \
    --stage1_top_k "$stage1_top_k" \
    --stage2_top_k "$stage2_top_k" \
    --agent_max_iters "$agent_max_iters" \
    --debug "$debug"