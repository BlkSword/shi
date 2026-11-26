"""远程编译并跑基准。"""
import os
import paramiko

HOST = os.environ.get("SHI_HOST", "103.242.14.227")
USER = os.environ.get("SHI_USER", "root")
PASS = os.environ.get("SHI_PASS", "")

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS, timeout=20, banner_timeout=20)

def run(cmd, timeout=1800):
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err

print("=== 编译 ===")
code, out, err = run(
    "source $HOME/.cargo/env && cd /opt/shi/rust && cargo build --release 2>&1 | tail -10",
    timeout=1200,
)
print(out)
if err:
    print("STDERR:", err[-500:])

print("=== 测试 ===")
code, out, err = run(
    "source $HOME/.cargo/env && cd /opt/shi/rust && cargo test --release 2>&1 | grep 'test result'",
    timeout=600,
)
print(out)

print("=== 基准 ===")
code, out, err = run("/opt/shi/rust/target/release/shi bench", timeout=600)
print(out)

print("=== 确定性验证 ===")
code, out, err = run("/opt/shi/rust/target/release/shi verify", timeout=600)
print(out)

c.close()
