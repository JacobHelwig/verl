# On-Policy Distillation

**Author:** [Jacob Helwig](https://jacobhelwig.github.io/)

Last updated: 05/14/2026.

---

> **Contents**
>
> - **Background** - Introduction to On-Policy Distillation
> - **Architecture** - Implementation and control flow of OPD 
> - **Usage** - Configuring OPD runs

---

## Background

### Summary

1. OPD is useful for distilling knowledge from teacher model(s) to a student model while aligning training states with inference states.
2. It is better than SFT/ vanilla KD because it is on-policy.
3. It is better than RLVR because it gives token-level, continuous rewards.

### Knowledge Distillation

KD is useful for distilling knowledge from a teacher model to a student model. Consider the task of solving math problems. Conventional KD will first sample reasoning traces + solution given math prompts from the teacher model, and then train the student model using a next-token prediction objectives to match the teacher logprobs.

Although effective, this introduces exposure bias: during training, the student will only learn how to act when it is in states sampled from the teacher. However, at inference time, states will be sampled from the student. Unless the student and teacher are perfectly aligned, the student will not have acquired knowledge from the teacher about how to act in its own state. In the example of math problems, perhaps the student distribution prefer algebraic proofs, but the teacher distribution prefers geometric proofs. While training distills knowledge primarily focused on the geometric proof, the student after training might still prefer algebraic proofs, thereby limiting the amount of knowledge acquired from the teacher that will be useful during rollouts.

### On-Policy RL

RLVR presents an alternative path: the student samples rollouts from its own rollout distribution, and if the rollout resulted in a correct solution, the logprobs of the overall solution are increased. This is good because now the training and inference distributions are aligned, however, the reward is binary and outcome based. That is, there is only 1 bit of information introduced into the system following a rollout, and it is at the rollout level, rather than more fine-grained and informative feedback at the token level.

### OPD

OPD [1,2,3] sits at the intersection of these two approaches. The student samples rollouts from its own distribution. Given the student rollout, the teacher returns logprobs of the next action in each of the student states. The student logprobs are then updated to match the teacher logprobs. In this way, the training and inference distribution are aligned, while the training signal is continuous and at the token level. Intuitively, this forces the teacher to provide knowledge to the student about how the teacher would act if it were in the student state. The teacher must provide the knowledge that it has given that the student has chosen to take the algebraic proof route.

More formally, OPD aims to minimize the following objective:

$$
E_{x\sim p_data, y \sim student distribution} \frac1{|y|}\sum_t^{|y|}D(\pi_{student}(\cdot|y_{<t}, x), \pi_{teacher}(\cdot|y_{<t}, x), y_t),
$$

where p_data is a distribution over prompts, \pi_student is the student model, \pi_teacher is the teacher model, and D is some divergence of estimator of divergence.

### Choice of divergence

[1] and [3] use two different types of divergence and optimization procedures, both of which we have implemented. [1], whose method we refer to as "GKD OPD" computes the full-vocab KL between the two distributions. Using forward KL, this is:

$$
D(\pi(\cdot), \nu(\cdot), y_t) = \sum_i^{|V|} \nu(y_i) \log\left(\frac\nu(y_i)\pi(y_i)\right).
$$

GKD then minimizes the distillation loss by directly backpropagating this divergence. 

*Note*: the current implementation (as of 5/14/26) only supports computing this over the top-k teacher logits. Thus, we only implement forward KL. In earlier implementations, we attempted to implement reverse KL using the student top-k, but found it to be unstable for 0.5B Qwen2.5. 

[3], whose method we refer to as "PG OPD" uses a negative reverse KL estimate as a reward for the model to maximize. In the original implementation of PG OPD, the k1 estimator of the reverse KL was used as

$$
D(\pi(\cdot), \nu(\cdot), y_t) = \text{sg}(\log\pi(y_t)-\log\nu(y_t)\right).
$$

Note that this Monte-Carlo estimator is only valid for reverse KL due to tokens being sampled from the student distribution. In order to get an estimator for forward KL, tokens would have to be sampled from the teacher. Additionally, the stop gradient is necessary because otherwise, when differentiating wrt \pi params, \nu will dissappear. 


### Multi-teacher OPD

MOPD is an exciting approach which has been adopted by several recent models as an alternative to RL. Most recently, several labs have used OPD with multiple domains [4,5,6,7]. The base model was RLed independently on diverse domains, such as math, coding, and instruction following. This produces an expert model for each domain. These experts were then used to OPD the student model on each of the domains together: given a mixture of math, coding, and IF data, the student is trained to match the logprobs of the respective teacher.


### Bibliography

[1] Agarwal, Rishabh, et al. "On-policy distillation of language models: Learning from self-generated mistakes." International Conference on Learning Representations. Vol. 2024. 2024.

[2] Yang, An, et al. "Qwen3 technical report." arXiv preprint arXiv:2505.09388 (2025).

[3] Lu, Kevin and Thinking Machines Lab, "On-Policy Distillation", Thinking Machines Lab: Connectionism, Oct 2025.

[4] Xiao, Bangjun, et al. "Mimo-v2-flash technical report." arXiv preprint arXiv:2601

[5] Zeng, Aohan, et al. "Glm-5: from vibe coding to agentic engineering." arXiv preprint arXiv:2602.15763 (2026).

[6] Yang, Zhuolin, et al. "Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation." arXiv preprint arXiv:2603.19220 (2026).

[7] DeepSeek-AI. "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence." (2026) 

## Architecture

OPD has two components, mirroring RL:

1. **Teacher logprob computation** — runs on a dedicated teacher resource pool
   (`distillation.n_gpus_per_node × distillation.nnodes`, allocated in
   [`verl/trainer/main_ppo.py`](../../verl/trainer/main_ppo.py)).
2. **Student optimization** — runs on the train workers, the same actor workers
   that handle PPO/GRPO updates.

### Teacher logprob computation

Teacher logprob computation is interleaved with rollouts inside the **Agent
Loop**. Each sample's teacher call fires as soon as its rollout finishes — there
is no batch-wide barrier — so teacher work overlaps with the still-running
rollouts on other samples.

#### Step-by-step

1. **Input.** `AgentLoopManager.generate_sequences(prompts: DataProto)` receives
   a batch of prompts
   ([`verl/experimental/agent_loop/agent_loop.py`](../../verl/experimental/agent_loop/agent_loop.py)).

2. **Chunking across workers.** The manager splits the batch evenly across its
   `AgentLoopWorker` actors:
   `chunks = prompts.chunk(len(self.agent_loop_workers))`, then dispatches each
   chunk via `worker.generate_sequences.remote(chunk)`
   ([`agent_loop.py`](../../verl/experimental/agent_loop/agent_loop.py)).

3. **Per-sample fan-out inside a worker.** Inside
   `AgentLoopWorker.generate_sequences`, each sample in the chunk is launched as
   its own asyncio task:
   `asyncio.create_task(self._run_agent_loop(...))`
   ([`agent_loop.py`](../../verl/experimental/agent_loop/agent_loop.py)).
   The agent loop runs on the rollout GPUs and produces a rollout (prompt +
   response token ids).

4. **Postprocess hook.** `_run_agent_loop` calls
   `self._agent_loop_postprocess(output, …)`
   ([`agent_loop.py`](../../verl/experimental/agent_loop/agent_loop.py)).
   This is where teacher logprob computation is triggered, per sample, as soon
   as that sample's rollout is ready.

5. **Worker-side teacher dispatch.** `_agent_loop_postprocess` calls
   `self._compute_teacher_logprobs(output, prompt_ids=…, response_ids=…, …)`
   ([`agent_loop.py`](../../verl/experimental/agent_loop/agent_loop.py)).
   This method extracts the routing value from the sample's non-tensor fields
   using
   `sample_kwargs[self.teacher_key]` (default `teacher_key = "data_source"`),
   then calls
   `self.teacher_server_manager.compute_teacher_logprobs_single(...)`.

6. **Teacher selection.**
   `AsyncTeacherLLMServerManager.compute_teacher_logprobs_single`
   ([`verl/experimental/teacher_loop/teacher_manager.py`](../../verl/experimental/teacher_loop/teacher_manager.py))
   resolves the teacher via `_resolve_teacher_key`:

   - **Single-teacher**: routing key is ignored; the sole configured teacher is
     used.
   - **Multi-teacher**: `routing_key` must match a configured teacher in
     `distillation.teacher_models`; otherwise an error is raised.

   The resolved key indexes into `self.teacher_client: dict[str, LLMServerClient]`
   to pick the right client.

7. **Sampling params for scoring (not generation).** The manager builds sampling
   params via `_get_teacher_sampling_params`
   ([`teacher_manager.py`](../../verl/experimental/teacher_loop/teacher_manager.py)):
   `max_tokens=1` plus `prompt_logprobs=topk` (or `0`) — the teacher *scores* the
   (prompt + response) sequence rather than generating new tokens. `topk` is
   set to `distillation.distillation_loss.topk` when the loss mode requires
   top-k (e.g. `forward_kl_topk`); otherwise `0` (single-sample logprob only).

8. **Server-side load balancing.** The manager calls `client.generate(...)`.
   Inside `LLMServerClient.generate`
   ([`verl/workers/rollout/llm_server.py`](../../verl/workers/rollout/llm_server.py)),
   the client acquires a backing server through the shared
   `GlobalRequestLoadBalancer` actor:

   - **Sticky session**: if the `request_id` was seen before and the previously
     chosen server is still in the pool, route to it (preserves vLLM prefix
     cache hits across multi-turn).
   - **Else least-loaded**: pick the server with the fewest in-flight requests.

9. **Backend execution.** With the vLLM backend, the selected server is a
   `vLLMHttpServer` actor
   ([`verl/workers/rollout/vllm_rollout/vllm_async_server.py`](../../verl/workers/rollout/vllm_rollout/vllm_async_server.py)).
   Its `generate` method runs the forward pass and returns a `TokenOutput`
   containing `prompt_ids`
   and `prompt_logprobs` for the full (prompt + response) sequence. The SGLang
   backend has an analogous server class.

10. **Return path.** `compute_teacher_logprobs_single` packs the response into
    two tensors of shape `(S, 1 or K)` — `teacher_ids` and `teacher_logprobs`,
    where `S` is the sequence length and `K = topk` (or `1`). These are stashed
    in `output.extra_fields["teacher_ids"]` / `["teacher_logprobs"]` and later
    concatenated into the per-batch `DataProto` in `_postprocess` for the
    student optimization step.


### Student Optimization

Using the `DataProto` produced by the Agent Loop (rollouts + teacher logprobs in
`teacher_ids` / `teacher_logprobs`), the student step proceeds as follows.

#### Step-by-step

1. **Train entry.** `TrainingWorker.train_batch`
   ([`verl/workers/engine_workers.py`](../../verl/workers/engine_workers.py))
   invokes `self.engine.train_batch(data, loss_function=self.loss_fn)`. When
   distillation is enabled, `self.loss_fn` is bound to
   `distillation_ppo_loss` at worker init
   (`partial(distillation_ppo_loss, config=actor_config, distillation_config=…)`);
   otherwise it is the standard `ppo_loss`.

2. **Forward pass and (optional) inline top-k loss.**
   `FSDPEngineWithLMHead.forward_step`
   ([`verl/workers/engine/fsdp/transformer_impl.py`](../../verl/workers/engine/fsdp/transformer_impl.py))
   runs the model forward, then calls `prepare_model_outputs(...,
   logits_processor_func=loss_function)`. If the active loss mode requires
   top-k (`distillation_use_topk=True`), `prepare_model_outputs` invokes
   `distillation_ppo_loss(student_logits=…, data=…)` **as a logits processor**
   while the full logits tensor is still in memory. This is the
   `student_logits is not None` branch in `distillation_ppo_loss`
   ([`verl/trainer/distillation/losses.py`](../../verl/trainer/distillation/losses.py)),
   which dispatches to a backend-specific `compute_forward_kl_topk` (FSDP /
   Megatron). Per-token `distillation_losses`, `student_mass`, and
   `teacher_mass` tensors are written back into `model_output` so the full
   logits can be freed before the final loss step.

3. **Final loss.** `forward_step` then calls `loss_function(model_output=…,
   data=…, dp_group=…)` — this is the `student_logits is None` branch of
   `distillation_ppo_loss`, where:

   1. **Per-token distillation loss** is produced by `distillation_loss(...)`,
      which dispatches via `get_distillation_loss_fn(loss_mode)` to one of
      two registered families
      ([`losses.py`](../../verl/trainer/distillation/losses.py)):

      - **Top-k** (`forward_kl_topk`, `use_topk=True`): reads the pre-computed
        per-token tensors from `model_output` (populated by the logits
        processor in step 2) and logs `student_mass` / `teacher_mass`
        diagnostics. Negative divergences (a top-k truncation artifact) are
        clamped to 0.
      - **Single-sample KL estimators** (`kl`, `k1`, `abs`, `mse`, `k2`,
        `low_var_kl`, `k3`, `use_estimator=True`): compares the student's
        per-token `log_probs` (from the forward pass) directly against the
        teacher's single log-prob in `data["teacher_logprobs"]` via
        `kl_penalty`. No logits-processor pass is needed.

   2. **Optional clamp.** If `loss_max_clamp` is set, per-token losses are
      clamped to `[-clamp, +clamp]` (k1 in particular can be negative).

   3. **Aggregation mode** — controlled by `use_policy_gradient`:

      - `False` (supervised): aggregate per-token losses via `agg_loss` over
        the response mask — straight backprop, as in
        [arxiv 2306.13649](https://arxiv.org/abs/2306.13649).
      - `True` (on-policy distillation): treat `-distillation_losses` as
        advantages and run PPO-style clipped importance sampling against
        `data["old_log_probs"]`, as in
        [Thinking Machines' on-policy distillation post](https://thinkingmachines.ai/blog/on-policy-distillation/).

   4. **Combine with task rewards.** A standard PPO policy loss is computed
      from the rollout's task rewards via `ppo_loss(...)`. If
      `use_task_rewards=False` it is zeroed; otherwise the final loss is
      `policy_loss + distillation_loss_coef * distill_loss`.

The returned scalar loss is what `engine.train_batch` backpropagates.