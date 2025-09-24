 #!/bin/bash
 vllm serve Alibaba-NLP/gme-Qwen2-VL-2B-Instruct --host 0.0.0.0 --port 8080 --device cuda --served-model-name gme 