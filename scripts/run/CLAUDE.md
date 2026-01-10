Hi Claude, we are in the verl repo. I would like to make a contribution and hope that you can help plan something to contribute. 

The most recent version of verl is 0.7.0. The version history in the file scripts/run/ROADMAP.md shows the changes in this version and changes that are planned for future versions. 

Please look through the roadmap, the code, and the commit history to help me identify some potential areas where I can contribute. Please create a file scripts/run/CONTRIBUTION_IDEAS.md and list some potential contribution ideas, along with a brief description of each idea and any relevant links to documentation or code.

Here is a deprecation map so that you can identify which files are deprecated and should not be worked on:

| File | Engine | Trainer |	Backend | Explaination |
| --- | --- | --- | --- | --- |
| fsdp_sft_trainer.py | legacy | SFT | fsdp | 
| fsdp_workers.py | legacy | RL | fsdp | 
| megatron_workers.py | legacy | RL | megatron | 
| sft_trainer.py | new | SFT | fsdp/megatron/veomni/... | torchrun spmd
| sft_trainer_ray.py | new | SFT | fsdp/megatron/veomni/... | ray single controller
| engine_workers.py | new | RL | fsdp/megatron/veomni/... |

Please do not suggest ideas for creating examples / adding documentation / adding tests. 