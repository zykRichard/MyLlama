# MyLlama
I implemented my own Llama model to learn the LLM architecture, referring to this helpful [tutorial](https://github.com/NJUDeepEngine/open-llm-assignments) 🤩.



⚠️ **MyLlama** has not been fully tested, its correctness is not guaranteed 

⚠️ I'm still completing some functions.



### TBD

- [ ] Sampling Module
- [ ] Simple Trainer & LoRA Trainer



### Module

|                  Module                   |                             File                             |                          Reference                           |
| :---------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                `RMS norm`                 |              [norm.py](./src/modeling/norm.py)               | [tutorial](https://github.com/NJUDeepEngine/open-llm-assignments/blob/a1-main/tasks/task1.md)<br />[Llama RMSnorm](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py#L60) |
|          `Vocabulary Embedding`           |         [vocab_emb.py](./src/modeling/vocab_emb.py)          | [tutorial](https://github.com/NJUDeepEngine/open-llm-assignments/blob/a1-main/tasks/task2.md)<br />[ChatGLM](https://huggingface.co/zai-org/chatglm3-6b/blob/main/modeling_chatglm.py#L710) |
|             `NTK-aware RoPE`              |           [pos_emb.py](./src/modeling/pos_emb.py)            | [tutorial](https://github.com/NJUDeepEngine/open-llm-assignments/blob/a1-main/tasks/task3.md)<br />[Llama RotaryEmb](https://github.com/huggingface/transformers/blob/v4.36.2/src/transformers/models/llama/modeling_llama.py#L178) |
| `Offline/Online Sliding Window Attention` |         [attention.py](./src/modeling/attention.py)          | [tutorial](https://github.com/NJUDeepEngine/open-llm-assignments/tree/a3-main)<br />[LlamaAttention](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py#L275)<br />[GQA](https://github.com/fkodom/grouped-query-attention-pytorch) |
|                 `MoE MLP`                 |               [mlp.py](./src/modeling/mlp.py)                | [tutorial](https://github.com/NJUDeepEngine/open-llm-assignments/tree/a2-main?tab=readme-ov-file)<br />[LlamaMLP](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py#L229)<br />[Mixtral MoE](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/mixtral/modeling_mixtral.py#L610) |
|            `KV Cache Manager`             | [TransformerDecoderKVCache](./src/modeling/transformer.py#L109) | [tutorial](https://github.com/NJUDeepEngine/open-llm-assignments/blob/a4-main/tasks/task1.md)<br />[HF Cache](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/cache_utils.py#L351)<br />[vLLM PageAttention](https://docs.vllm.ai/en/latest/design/paged_attention.html) |

