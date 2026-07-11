# CCI persistent-storage requirement

This repository is developed and run inside an ephemeral CCI GPU container.
The container's `/home` directory is reset whenever the container restarts.

- Store all mutable or recoverable artifacts under `/data/L202500291`, never
  under `/home`. This includes Python/Conda environments, package and model
  caches, Codex state, Docker image archives or data roots, Ray temporary
  files, datasets, checkpoints, logs, and experiment outputs.
- Treat `/home` only as a bootstrap location for minimal shell configuration or
  symlinks that can be restored from a fixed base image.
- Prefer configurations and launch scripts that recreate those `/home` links
  automatically while keeping all actual state in `/data/L202500291`.

# Long-running commands

Background jobs started with `nohup ... &` from the tool shell can be killed
when that shell exits. Do not use `nohup` for training, inference, downloads,
or other long-running jobs.

- Launch long-running work in a long-lived `exec_command` session instead.
- Keep that session alive by polling it with `write_stdin` and pipe output
  through `tee` when logs are needed.
- If the session is interrupted, inspect active processes with `pgrep`, GPU
  state with `nvidia-smi`, and existing output line counts before resuming.
