# SimpleTIR local code-execution profile. Installed by start_local_sandbox.sh.
# The API passes --private=<per-request-workdir>; keep this profile independent
# of the container's ephemeral /home state.
net none
private-dev
rlimit as 512M
rlimit cpu 3
rlimit nproc 50
caps.drop all
seccomp
nonewprivs
blacklist /data
blacklist /home
blacklist /root
whitelist /usr/bin/python3
include /etc/firejail/whitelist-common.inc
