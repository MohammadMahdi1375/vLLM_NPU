# vLLM_NPU spec_main workspace

This branch tracks the official vLLM-related repositories as Git submodules:

- vllm: https://github.com/vllm-project/vllm.git
- vllm-ascend: https://github.com/vllm-project/vllm-ascend.git
- speculators: https://github.com/vllm-project/speculators.git

Update all official sources with:

git submodule update --remote --merge --recursive
git add vllm vllm-ascend speculators
git commit -m "Update official submodule versions"
git push
