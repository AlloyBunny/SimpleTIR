# Multi-turn 3B full-parameter GRPO design

## Goal

Run the repository's `simpletir_trainer` as an end-to-end multi-turn tool-use
GRPO job with Qwen2.5-3B-Instruct.  The job trains every model parameter, uses
both provided training datasets, and stores all mutable state under
`/data/L202500291`.

This is not an SFT cold-start phase and it does not add LoRA support.

## Constraints and findings

- The repository's supported multi-turn recipe uses `agent.tool_use=True`,
  five turns, vLLM rollout, and an HTTP sandbox endpoint.
- The CCI container has four H100 80 GB GPUs, whereas the README's example
  uses eight H100 GPUs.  The job therefore cannot claim to reproduce the
  README's throughput or exact global-batch configuration.
- Firejail can start in this container but does not hide `/data` because the
  container lacks `CAP_SYS_ADMIN`.  It must not execute untrusted model code.
- An unprivileged user namespace, including mount, network, and PID namespaces,
  is available to the `sandboxer` account.  Bubblewrap can use that kernel
  primitive without privileged container capabilities.
- The existing GRPO FSDP-to-vLLM synchronization sends the full actor state
  dictionary into the rollout engine.  It has no LoRA adapter synchronization
  path, so this design deliberately uses full-parameter training.

## Sandbox architecture

The local HTTP interface remains compatible with the existing
`SANDBOX_ENDPOINT` client.  The execution backend changes from Firejail to
Bubblewrap.

1. The bootstrap script installs Bubblewrap if absent, then starts the FastAPI
   service as the unprivileged `sandboxer` account on `127.0.0.1`.
2. Each request starts a fresh Bubblewrap process with a new user, mount,
   network, PID, IPC, and UTS namespace.  It runs as UID 0 only *inside* the
   new user namespace; on the host it remains `sandboxer`.
3. The sandbox exposes only read-only runtime directories required by the
   system Python (`/usr`, `/bin`, `/lib`, and `/lib64` when present), an empty
   `/tmp`, a minimal `/dev`, a private `/proc`, and the per-request work
   directory.  It does not bind `/data`, `/home`, `/root`, or the repository.
4. The network namespace contains no host network interfaces.  The server also
   keeps CPU, address-space, process-count, and wall-clock limits for each
   request.
5. Startup is fail-closed: it tests that code can run, cannot see a known
   `/data` path, and cannot reach the network.  If any check fails, the server
   is stopped and training cannot start.

The Bubblewrap package is installed in the ephemeral base container, but all
scripts, logs, PID files, temporary service state, models, checkpoints, Ray
state, and caches remain under `/data/L202500291`.  Bootstrap makes the
ephemeral package installation repeatable after a container restart.

## Training architecture

1. Download the public Qwen2.5-3B-Instruct checkpoint into a new persistent
   model directory under `/data/L202500291` and verify its tokenizer and weight
   files before launch.
2. Add one explicit launch script for this experiment.  It sources the
   persistent environment setup, requires the local sandbox readiness check,
   exports `SANDBOX_ENDPOINT`, and invokes `train.sh` with
   `CONFIG_NAME=simpletir_trainer`, `tool_use=True`, `max_turns=5`, and both
   `simplelr_math_35/train` and `deepscaler/train`.
3. Use the normal FSDP actor and vLLM rollout path.  No model, optimizer, or
   GRPO implementation is patched.
4. Keep the entire dataset and all model parameters trainable.  Scale only
   hardware-dependent settings (global batch, PPO mini-batch, rollout tensor
   parallelism, vLLM memory utilization, and possibly rollout count) after a
   capacity probe on the four available GPUs.  The probe must keep tool use,
   the reward manager, and the five-turn agent enabled.
5. Put checkpoints, logs, generated outputs, Ray temporary directories, and
   Hugging Face caches under `/data/L202500291/outputs/simpletir` or the
   existing persistent cache root.  Enable a documented resume path.

## Validation and failure handling

Before a long-running job, validation proceeds in this order:

1. Unit-test the Bubblewrap command construction and fail-closed startup.
2. Start the service and prove via HTTP that basic Python works, `/data` is not
   visible, and the network is unavailable.
3. Run one real `simpletir_trainer` optimization step with the production
   multi-turn tool-use path and persistent outputs.  Confirm a successful code
   execution, non-filtered reward processing, an optimizer update, checkpoint
   creation, and cleanup of GPUs and Ray processes.
4. Select the largest stable four-GPU configuration from the observed memory
   and throughput, then launch the resumable full-data job.

If sandbox isolation fails, the launcher exits before any model-generated code
is executed.  If the capacity probe exhausts GPU memory, only the
hardware-dependent parameters are reduced; the training method remains
full-parameter multi-turn GRPO.

## Non-goals

- Reproducing the README's exact eight-H100 batch size or wall-clock throughput.
- Running unsafe code through Firejail's ineffective mount isolation.
- Adding LoRA/PEFT support to the GRPO path.
- Creating or training on an artificial SFT dataset.
