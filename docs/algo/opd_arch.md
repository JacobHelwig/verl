# On-Policy Distillation Architecture

OPD has two components, mirroring RL:

1. **Teacher logprob computation** — runs on a dedicated teacher resource pool
   (`distillation.n_gpus_per_node × distillation.nnodes`, allocated in
   [`verl/trainer/main_ppo.py:174–182`](../../verl/trainer/main_ppo.py)).
2. **Student optimization** — runs on the train workers, the same actor workers
   that handle PPO/GRPO updates.

This document describes component (1).

## Teacher logprob computation

Teacher logprob computation is interleaved with rollouts inside the **Agent
Loop**. Each sample's teacher call fires as soon as its rollout finishes — there
is no batch-wide barrier — so teacher work overlaps with the still-running
rollouts on other samples.

### Step-by-step

1. **Input.** `AgentLoopManager.generate_sequences(prompts: DataProto)` receives
   a batch of prompts
   ([`verl/experimental/agent_loop/agent_loop.py:1069`](../../verl/experimental/agent_loop/agent_loop.py)).

2. **Chunking across workers.** The manager splits the batch evenly across its
   `AgentLoopWorker` actors:
   `chunks = prompts.chunk(len(self.agent_loop_workers))`, then dispatches each
   chunk via `worker.generate_sequences.remote(chunk)`
   ([`agent_loop.py:1078–1083`](../../verl/experimental/agent_loop/agent_loop.py)).

3. **Per-sample fan-out inside a worker.** Inside
   `AgentLoopWorker.generate_sequences`, each sample in the chunk is launched as
   its own asyncio task:
   `asyncio.create_task(self._run_agent_loop(...))`
   ([`agent_loop.py:526–537`](../../verl/experimental/agent_loop/agent_loop.py)).
   The agent loop runs on the rollout GPUs and produces a rollout (prompt +
   response token ids).

4. **Postprocess hook.** `_run_agent_loop` calls
   `self._agent_loop_postprocess(output, …)`
   ([`agent_loop.py:577`](../../verl/experimental/agent_loop/agent_loop.py)).
   This is where teacher logprob computation is triggered, per sample, as soon
   as that sample's rollout is ready.

5. **Worker-side teacher dispatch.** `_agent_loop_postprocess` calls
   `self._compute_teacher_logprobs(output, prompt_ids=…, response_ids=…, …)`
   ([`agent_loop.py:686`](../../verl/experimental/agent_loop/agent_loop.py)).
   This method, defined at
   [`agent_loop.py:870–893`](../../verl/experimental/agent_loop/agent_loop.py),
   extracts the routing value from the sample's non-tensor fields using
   `sample_kwargs[self.teacher_key]` (default `teacher_key = "data_source"`),
   then calls
   `self.teacher_server_manager.compute_teacher_logprobs_single(...)`.

6. **Teacher selection.**
   `AsyncTeacherLLMServerManager.compute_teacher_logprobs_single`
   ([`verl/experimental/teacher_loop/teacher_manager.py:102–128`](../../verl/experimental/teacher_loop/teacher_manager.py))
   resolves the teacher via `_resolve_teacher_key`
   ([`teacher_manager.py:86–100`](../../verl/experimental/teacher_loop/teacher_manager.py)):

   - **Single-teacher**: routing key is ignored; the sole configured teacher is
     used.
   - **Multi-teacher**: `routing_key` must match a configured teacher in
     `distillation.teacher_models`; otherwise an error is raised.

   The resolved key indexes into `self.teacher_client: dict[str, LLMServerClient]`
   to pick the right client.

7. **Sampling params for scoring (not generation).** The manager builds sampling
   params via `_get_teacher_sampling_params`
   ([`teacher_manager.py:30–43`](../../verl/experimental/teacher_loop/teacher_manager.py)):
   `max_tokens=1` plus `prompt_logprobs=topk` (or `0`) — the teacher *scores* the
   (prompt + response) sequence rather than generating new tokens. `topk` is
   set to `distillation.distillation_loss.topk` when the loss mode requires
   top-k (e.g. `forward_kl_topk`); otherwise `0` (single-sample logprob only).

8. **Server-side load balancing.** The manager calls `client.generate(...)`
   ([`teacher_manager.py:114`](../../verl/experimental/teacher_loop/teacher_manager.py)).
   Inside `LLMServerClient.generate`
   ([`verl/workers/rollout/llm_server.py:179–219`](../../verl/workers/rollout/llm_server.py)),
   the client acquires a backing server through the shared
   `GlobalRequestLoadBalancer` actor:

   - **Sticky session**: if the `request_id` was seen before and the previously
     chosen server is still in the pool, route to it (preserves vLLM prefix
     cache hits across multi-turn).
   - **Else least-loaded**: pick the server with the fewest in-flight requests
     ([`llm_server.py:68–91`](../../verl/workers/rollout/llm_server.py)).

9. **Backend execution.** With the vLLM backend, the selected server is a
   `vLLMHttpServer` actor
   ([`verl/workers/rollout/vllm_rollout/vllm_async_server.py:81`](../../verl/workers/rollout/vllm_rollout/vllm_async_server.py)).
   Its `generate` method
   ([`vllm_async_server.py:447`](../../verl/workers/rollout/vllm_rollout/vllm_async_server.py))
   runs the forward pass and returns a `TokenOutput` containing `prompt_ids`
   and `prompt_logprobs` for the full (prompt + response) sequence. The SGLang
   backend has an analogous server class.

10. **Return path.** `compute_teacher_logprobs_single` packs the response into
    two tensors of shape `(S, 1 or K)` — `teacher_ids` and `teacher_logprobs`,
    where `S` is the sequence length and `K = topk` (or `1`)
    ([`teacher_manager.py:123–128`](../../verl/experimental/teacher_loop/teacher_manager.py)).
    These are stashed in `output.extra_fields["teacher_ids"]` /
    `["teacher_logprobs"]`
    ([`agent_loop.py:892–893`](../../verl/experimental/agent_loop/agent_loop.py))
    and later concatenated into the per-batch `DataProto` in `_postprocess`
    ([`agent_loop.py:914–916`](../../verl/experimental/agent_loop/agent_loop.py))
    for the student optimization step.
