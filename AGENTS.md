# CCI persistent-storage requirement

This repository is developed and run inside an ephemeral CCI GPU container.
The container's `/home` directory is reset whenever the container is restarted.

- Store all mutable or recoverable artifacts under `/data/L202500291`, never under `/home`.
- This includes Python/Conda environments, package and model caches, Codex state,
  Docker image archives or Docker data roots, Ray temporary files, datasets,
  checkpoints, logs, and experiment outputs.
- Treat `/home` only as a bootstrap location for minimal shell configuration or
  symlinks that can be restored from a fixed base image.
- Prefer configurations and launch scripts that recreate those `/home` links
  automatically while keeping all actual state in `/data/L202500291`.
