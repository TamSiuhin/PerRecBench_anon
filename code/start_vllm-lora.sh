SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${PERRECBENCH_MODEL_DIR:-${SCRIPT_DIR}/../sft/model}"

CUDA_VISIBLE_DEVICES=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
--api_key EMPTY \
--enable-lora \
--lora-modules perrecbench-lora-mix="${MODEL_DIR}/PerRecBench-llama-3.1-8b-sft-lora-mix" perrecbench-lora-avg="${MODEL_DIR}/PerRecBench-llama-3.1-8b-sft-lora-avg" perrecbench-lora-point="${MODEL_DIR}/PerRecBench-llama-3.1-8b-sft-lora-point" perrecbench-lora-group="${MODEL_DIR}/PerRecBench-llama-3.1-8b-sft-lora-group" perrecbench-lora-pair="${MODEL_DIR}/PerRecBench-llama-3.1-8b-sft-lora-pair" \
--served-model-name PerRecBench-llama-3.1-8b-sft-lora \
--tensor-parallel-size 1 \
--port 8002 \
--max-model-len 40000
