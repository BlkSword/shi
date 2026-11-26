"""在远程安装 Rust 并编译。"""
import os
import paramiko

HOST = os.environ.get("SHI_HOST", "103.242.14.227")
USER = os.environ.get("SHI_USER", "root")
PASS = os.environ.get("SHI_PASS", "")

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS, timeout=20, banner_timeout=20)

def run(cmd, timeout=900):
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err

print("=== 安装 Rust ===")
code, out, err = run(
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal 2>&1 | tail -5",
    timeout=600,
)
print(out[-800:])
if err:
    print("STDERR:", err[-500:])

print("=== 验证 ===")
code, out, err = run("source $HOME/.cargo/env && cargo --version && rustc --version")
print(out)

c.close()
