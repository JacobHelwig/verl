There are 2 components in OPD, same as RL. The first is teacher logprob computation. This happens on a set of dedicated teacher resources. The second is student optimization. This happens on the train worker.

## Teacher logprob computation

This happens in the Agent Loop, where we stream teacher logprob computation with rollouts.


1. Batch data: a batch of prompts

2. Batch data is chunked between the AgentLoopWorker; for each prompt in chunk, agent loop is run on rollout GPUs, producing a rollout 

3. As part of the AgentLoopWorker._run_agent_loop, we call _agent_loop_postprocess. This is where teacher logprob computation happens in a streaming manner

4. Here we call AsyncTeacherLLMServerManager._compute_teacher_logprobs. This builds the sampling parameters needed for the logprob computation on the teacher server, and selects a client of type LLMServerClient to send the request via LLMServerClient.generate. 

5. The teacher client is selected in AsyncTeacherLLMServerManager._compute_teacher_logprobs according to the example; if we are not in multi-teacher mode, there is only one teacher client, but otherwise, the example will have a key mapping us to the appropriate teacher client 

6. The client will select one of the teacher servers based on load balancing criterion established by GlobalRequestLoadBalancer. If using vLLM inference backend, this will be vLLMHttpServer

7. Teacher logprobs are computed and returned in the servers's generate method