# Entropy memory investigation: run 3661417

The run failed in `_lm_head_logits_processor -> vocab_parallel_entropy ->
_VocabParallelEntropy.forward` during old-policy log-prob evaluation. CUDA
reported a 9.47 GiB allocation with only 2.50 GiB free. This followed the earlier
host-memory failure in run 3660836; requesting 850,000 MiB per node addressed
the allocation limit on host RAM but did not remove this GPU temporary.

## Source evidence

- `verl/verl/trainer/ppo/v1/trainer_base.py::_compute_old_log_prob` always
  requests entropy, stores it in TransferQueue, and reports `actor/entropy`.
- The configured entropy loss coefficient is zero. Actor updates ordinarily
  request entropy only when explicitly enabled or its coefficient is nonzero.
- `verl/verl/workers/engine/base.py::infer_batch` executes the log-prob forward
  under `torch.no_grad()`.
- The deployed `_lm_head_logits_processor` already supports
  `entropy_from_logits_with_chunking` and `entropy_from_logits_chunk_size`.
- The existing chunked routine calls the same tensor-parallel entropy function
  for each sequence-token chunk. The current recipe uses padded BSHD tensors,
  matching that implementation's sequence-axis slicing.

## Minimal configuration change

```text
actor_rollout_ref.actor.megatron.entropy_from_logits_with_chunking=True
actor_rollout_ref.actor.megatron.entropy_from_logits_chunk_size=1024
```

At roughly 40,960 sequence tokens, the failed 9.47 GiB temporary would scale to
about 0.24 GiB per 1,024-token chunk. This is an estimate for that temporary,
not a claim about total GPU peak usage. The existing full logits and protective
clone still exist. Forward-only execution avoids retaining autograd state for
all chunks. The installed Megatron cross-entropy source also uses in-place
normalization/exponentiation, so it does not necessarily introduce another
temporary of the same size at the next call.

This preserves the entropy/log-prob formulas and avoids changing sequence
lengths, parallelism, PPO settings or the logged entropy metric. No third-party
kernel or optimizer code was edited. Lint/shell/source checks cover the config
change; numerical parity and successful training have not been demonstrated by
this investigation. The next run's telemetry provides the runtime check.
