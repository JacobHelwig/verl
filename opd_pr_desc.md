
# What does this PR do?

Adds on-policy distillation support across FSDP and Megatron backends.

Collaboration with @wuxibin89, including design guidance, restructuring, and added support for parallelism.

Supports: 

- FSDP and Megatron engines 
- top-k distillation loss and KL estimator distillation losses 
- Supervised and policy-gradient-style updates
- Teacher logprobs computation using a vLLM teacher server
- LLM and VLM distillation
- FSDP sequence parallel
- Megatron context parallel and tensor parallel

## Losses 

1. top-k distillation loss: **forward** KL estimated using top-k logits **from teacher**. 
2. KL estimator distillation losses: **reverse** KL estimated using only the log prob for the sampled token via the same estimators used by the reference model (e.g., k1, k3)  

## Updates

1. Supervised: distillation loss is directly backpropagated, as in https://arxiv.org/abs/2306.13649
2. Policy gradient: negative distillation loss is used as a reward, as in https://thinkingmachines.ai/blog/on-policy-distillation/

# Test

- LLM distillation with FSDP: `examples/on_policy_distillation_trainer/run_qwen_gsmk8k.sh`
- VLM distillation with FSDP: `examples/on_policy_distillation_trainer/run_qwen3_vl_geo3k.sh`
- LLM distillation with megatron: `examples/on_policy_distillation_trainer/run_qwen_gsmk8k_megatron.sh`. 


## Main results 

### LLM Distillation 

These experiments compare 3 training runs with student model Qwen2.5-0.5B using `examples/on_policy_distillation_trainer/run_qwen_gsmk8k.sh`:

1. Forward top-k KL with Qwen2.5-3B-Instruct teacher (gold)
2. Forward top-k KL with Qwen2.5-7B-Instruct teacher (green)
3. k3 estimator KL with Qwen2.5-7B-Instruct teacher (red)

#### GSM8K eval acc

<img width="969" height="641" alt="image" src="https://github.com/user-attachments/assets/19c1bee7-b688-4d24-a41e-4426761a26f1" />

#### GSM8K train acc

<img width="972" height="639" alt="image" src="https://github.com/user-attachments/assets/8f932649-5d45-4964-9d59-18a3706004d5" />

#### Distillation loss

<img width="963" height="633" alt="image" src="https://github.com/user-attachments/assets/609f817b-2247-42df-86a8-5e07d637ea7c" />

### VLM Distillation 

- Data: Geometry3K
- Student: Qwen3-VL-2B-Instruct
- Teacher: Qwen3-VL-4B-Instruct
- OPD algo: k1 KL estimator as reward with policy gradient loss

#### Geo3K eval acc

<img width="967" height="640" alt="image" src="https://github.com/user-attachments/assets/e511fedb-8cf1-4576-b214-e992984c7550" />

#### Geo3K train acc

<img width="963" height="630" alt="image" src="https://github.com/user-attachments/assets/0eb0e281-218d-4283-825e-2d0b1aa095d4" />

#### Distillation loss

<img width="964" height="633" alt="image" src="https://github.com/user-attachments/assets/7b539c60-2af8-4d30-a9d3-3cb4cf485847" />

## LLM Distillation: Top-k training stability

Clamping the top-k forward KL loss was needed for training stability. These experiments compare 3 types of clamping: 

1. No clamping (grey)
2. Clamping the distillation loss to a maximum value of 10 (blue)
3. Clamping the student and teacher log probs to a minimum value of -10 (gold)

### Distillation loss

<img width="947" height="633" alt="image" src="https://github.com/user-attachments/assets/e90815f5-f745-4a07-bc41-dbb3eb16b1dc" />

### GSM8K eval acc

<img width="971" height="639" alt="image" src="https://github.com/user-attachments/assets/d5cd1833-7e66-40f3-b0a3-fae6a9bc0d7b" />

### GSM8K train acc

<img width="958" height="636" alt="image" src="https://github.com/user-attachments/assets/53742e8d-cd41-4483-943e-cea2b5f0c27b" />

## LLM Distillation: Policy-gradient results

While the VLM results in this PR use the k1 KL estimator with policy gradient updates, all LLM distillation results outside of this section rely on supervised updates. LLM distillation with policy gradient updates are validated in this section:

1. Forward top-k KL with supervised update (green)
2. k1 estimator KL with policy gradient update (purple)
3. k3 estimator KL with supervised update (red)

While purple seems best, it also is generating responses that exceed the maximum response length of 512.

### Distillation loss

<img width="971" height="627" alt="image" src="https://github.com/user-attachments/assets/208b8625-23d7-46d5-982d-6ab1b5049b21" />


### GSM8K eval acc

<img width="976" height="632" alt="image" src="https://github.com/user-attachments/assets/dd46c6d3-651a-45cb-8bc5-75a765a9a38e" />


### GSM8K train acc

<img width="969" height="634" alt="image" src="https://github.com/user-attachments/assets/f2db7e16-5426-4fa2-b017-fb78b58b8dd4" />

### Response length

<img width="979" height="640" alt="image" src="https://github.com/user-attachments/assets/f61fadd7-d1a8-4110-ba26-a5a2735f8107" />

## LLM Distillation: Megatron 

To verify parity of megatron engine with FSDP, these experiments compare 3 training runs with student model Qwen2.5-0.5B:

1. Forward top-k KL with Qwen2.5-7B-Instruct teacher + clamping log probs to minimum value of -10.0 (teal)
2. Forward top-k KL with Qwen2.5-3B-Instruct teacher + clamping log probs to minimum value of -10.0 (red)
3. Forward top-k KL with Qwen2.5-3B-Instruct teacher + clamping loss to maximum value of 10.0 (blue)
4. k3 estimator reverse KL with Qwen2.5-7B-Instruct teacher + clamping loss to maximum value of 10.0 (green)

The solid line uses megatron engine with TP=2, the dotted line uses FSDP.

### GSM8K Eval Acc.

<img width="1656" height="839" alt="image" src="https://github.com/user-attachments/assets/e43a31a9-f012-452f-b606-99ede49a5fce" />

### GSM8K Train Acc.

<img width="1653" height="838" alt="image" src="https://github.com/user-attachments/assets/609b5917-dbb4-4a72-bbb8-acd4a7f4224a" />

### Distillation Loss

<img width="1651" height="846" alt="image" src="https://github.com/user-attachments/assets/6842295a-aa9a-4b11-9153-b9f83130e352" />


### Grad Norm

<img width="988" height="636" alt="image" src="https://github.com/user-attachments/assets/5883e67c-71e3-4e65-87e3-79e0a2527757" /> 

## LLM Distillation: Note on reverse KL

Initially, this PR included top-k reverse KL and top-k Jensen-Shannon divergences (JSD interpolates between forward and reverse KL). For the student distribution $q$ and teacher distribution $p$, the top-k reverse KL is given by

$$
KL_{\text{top-}k}(q||p) = \sum_i \bf{1}(q_i\in \text{top-}k)q_i\log\frac{q_i}{p_i}.
$$ 

Unfortunately, this was unstable. The reason is because one way to make this loss small is to make $q_i$ as small as possible for all $q_i \in \text{top}-k$. This can be seen from the logs tracking the amount of mass captured in the top-$k$ probabilities:

<img width="1118" height="1286" alt="image" src="https://github.com/user-attachments/assets/6e335263-7f4d-48d1-96f4-181f89b24e21" />

## LLM Distillation: Ablation: performance with more lenient parser

Note that the only loss used is the distillation loss (no rewards for correctness on GSM8K). Any increase in the logged rewards=GSM8k accuracy are an indirect result of minimizing the distillation loss. The reason that the base model has Pass@1~=0 is because the default GSM8k answer formatting (`#### 42`) is OOD for the model. The base model is answering the questions correctly, but using incorrect formatting, so none of the answers can be parsed. The base model can be evaluated using a reward function that is more lenient on formatting by adding the following to the script:

```bash
...
    reward_model.reward_manager=remote \
    custom_reward_function.path=tests/experimental/reward_loop/reward_fn.py \
    custom_reward_function.name=compute_score_math_verify \
    trainer.val_only=True
``` 

The results are:

```bash
(TaskRunner pid=904198) ("Initial validation metrics: {'val-aux/openai/gsm8k/reward/mean@1': "
(TaskRunner pid=904198)  "np.float64(0.31766489764973466), 'val-core/openai/gsm8k/acc/mean@1': "
(TaskRunner pid=904198)  "np.float64(0.31766489764973466), 'val-aux/num_turns/min': np.int32(2), "
(TaskRunner pid=904198)  "'val-aux/num_turns/max': np.int32(2), 'val-aux/num_turns/mean': "
(TaskRunner pid=904198)  'np.float64(2.0)}')
```

# Design & Code Changes

- Teacher workers are used in the agent loop, similar to the generative reward model: after a student worker finishes its rollout, the teacher worker obtains logprobs
- In the initial version of this PR (#4897), requests were submitted to the `vLLMHttpServer` via the `v1/completions` endpoint, which does not support multi-modal data. While `v1/chat/completions` does support multi-modal inputs, text must be passed as raw text instead of token IDs, preventing exact scoring of student generations since `student gen IDs -> student gen text -> teacher input IDs via v1/chat/completions tokenization` will not always give `student gen IDs == teacher input IDs` (https://vllm.ai/blog/agent-lightning). 
- This PR instead follows a path similar to how rollout replicas directly call the `generate` method on the `vLLMHttpServer`. This enables multi-modal inputs while representing text as token IDs. Requests to the teacher server now call the newly-added `compute_logprobs` method of `vLLMHttpServer`. 



# Prompt

 
hi codex, I hope that we can add docs for on-policy distillation implementation. There are a few things to do before we do this:
 
1. Review the PR. Its checked into main with hash 455e44c6feeabe378ad9dcc790fe9313926de12c, and the PR description itself will be very useful for our writing. I added the PR description to opd_pr_desc.md. Please read the PR desc and examine the diff to understand the scope of the changes

2. Please add a markdown doc summarizing the changes made in this PR

3. Please find where we should add the doc. Personally, I think the best place to it will be under docs/algo, following the style of docs/algo/grpo.md

4. After finding a good location and a reference doc, please create the OPD doc by following the style of the reference doc.  



# Doc visualization env

```bash
conda create -n verl_docs python=3.10 -y
conda activate verl_docs
uv pip install -r requirements-docs.txt
```

# Spin up docs

```python
make clean 
make html
python -m http.server -d _build/html/ 8006
```