# Local Sandbox and Smoke Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Document the persistent SimpleTIR environment, provide a recoverable local Firejail service, and run a minimal 4-GPU evaluation.

**Architecture:** A repository-stored Firejail profile is copied to its required system location by a root-only launcher. The launcher runs the existing FastAPI implementation as a non-login user on `127.0.0.1`, while all logs and lifecycle files live under `/data/L202500291/sandbox`. The smoke command uses the existing Qwen2.5 7B checkpoint and project datasets.

**Tech Stack:** Bash, Firejail, FastAPI, Uvicorn, Pytest, PyTorch, vLLM, Ray.

## Global Constraints

- Keep all recoverable state under `/data/L202500291`; `/etc/firejail/sandbox.profile` is recreated from a repository copy.
- Bind the sandbox to `127.0.0.1` only and execute snippets as the `sandboxer` non-login user.
- Use `/data/L202500291/miniconda3/envs/simpletir/bin/python` through `scripts/activate_simpletir_env.sh`.
- Do not alter the existing model, dataset, or user-authored unrelated changes.

---

### Task 1: Record the verified runtime configuration

**Files:**
- Modify: `文档/2026-07-11-CCI-环境启动清单.md`

**Interfaces:**
- Consumes: verified environment versions, model path, data paths, and the launcher command.
- Produces: a restart-safe operator guide.

- [ ] Add verified package versions, activation command, final verification-log path, and persistent cache rules.
- [ ] Add the Qwen2.5-7B-Instruct-1M path and the five packaged parquet datasets with row counts.
- [ ] Add the local sandbox lifecycle commands and a one-sample 4-GPU smoke-evaluation command.
- [ ] Check the document contains no obsolete claim that the `simpletir` environment is uncreated.

### Task 2: Add a recoverable local sandbox launcher

**Files:**
- Create: `sandbox/firejail/sandbox.profile`
- Create: `scripts/start_local_sandbox.sh`
- Create: `scripts/stop_local_sandbox.sh`
- Create: `tests/sandbox/test_local_sandbox_setup.py`

**Interfaces:**
- Consumes: `scripts/activate_simpletir_env.sh` and the existing `sandbox/sandbox_api.py` FastAPI app.
- Produces: a background service at `http://127.0.0.1:12345/faas/sandbox/`, with PID and log files in `/data/L202500291/sandbox`.

- [ ] Write a test that requires a launcher, stop script, and profile with loopback binding, persistent runtime directory, non-login user, and Firejail profile installation.
- [ ] Run `pytest tests/sandbox/test_local_sandbox_setup.py -q` and confirm it fails because the launcher artifacts do not exist.
- [ ] Add the profile and scripts with the tested interface; the launcher must create required directories, install the profile, create `sandboxer`, and wait for `/docs` to respond before reporting the endpoint.
- [ ] Re-run `pytest tests/sandbox/test_local_sandbox_setup.py -q` and `bash -n` on both scripts.

### Task 3: Provision and validate the sandbox service

**Files:**
- Runtime: `/etc/firejail/sandbox.profile`
- Runtime: `/data/L202500291/sandbox/local-sandbox.log`
- Runtime: `/data/L202500291/sandbox/local-sandbox.pid`

**Interfaces:**
- Consumes: `scripts/start_local_sandbox.sh`.
- Produces: a functioning endpoint accepted by `sandbox/local_sandbox.py`.

- [ ] Install `firejail` and create the isolated `sandboxer` system user through the launcher.
- [ ] Start the service and query `http://127.0.0.1:12345/docs`.
- [ ] POST `print(1 + 1)` and assert a success response containing `2`.
- [ ] Run `SANDBOX_ENDPOINT=http://127.0.0.1:12345/faas/sandbox/ python sandbox/local_sandbox.py` to check the client contract.

**CCI result:** Firejail was installed and the service lifecycle was validated, but the current
container lacks `CAP_SYS_ADMIN` for nested mount namespaces. Its fallback path can execute code
without isolation, which the launcher now detects by asserting `/data` is inaccessible before
declaring readiness. The launcher correctly exits with status `1`; a compatible remote endpoint
or a privileged container is required for this task.

### Task 4: Attempt the minimal SimpleTIR reproduction

**Files:**
- Runtime: `/data/L202500291/outputs/simpletir/logs/`
- Runtime: `/data/L202500291/outputs/simpletir/checkpoints/`

**Interfaces:**
- Consumes: the verified Qwen2.5 7B model, packaged `deepscaler/aime` data, and local sandbox endpoint.
- Produces: a one-sample, one-turn, 4-GPU `val_only` run or an evidence-backed incompatibility record.

- [ ] Source the activation script and set persistent model, data, output, W&B offline, GPU, and sandbox variables.
- [ ] Run `train.sh` with `--model_name Qwen2.5-7B-Instruct-1M`, `--valid_dataset deepscaler/aime`, `--val_only True`, `--max_turns 1`, `--max_prompt_length 4096`, `--max_response_length 1024`, `--val_sample_size 1`, `--n_val 1`, `--rollout_tp 2`, and `--sp_size 1`.
- [ ] Record its exit status and the generated log path in the operator guide or final handoff; do not modify unrelated project code to mask an upstream failure.
