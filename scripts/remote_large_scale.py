"""部署并运行大规模测试。"""
import os
import tarfile
import io
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

print("=== 上传源码 ===")
sftp = c.open_sftp()
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    tar.add("rust", arcname="rust")
buf.seek(0)
sftp.putfo(buf, "/tmp/shi-rust.tar.gz")
print("  uploaded", buf.tell(), "bytes")
sftp.close()

code, out, err = run("rm -rf /opt/shi/rust && cd /opt/shi && tar xzf /tmp/shi-rust.tar.gz && ls rust/")
print(out)

print("=== 编译 ===")
code, out, err = run(
    "source $HOME/.cargo/env && cd /opt/shi/rust && cargo build --release 2>&1 | tail -5",
    timeout=1200,
)
print(out)

print("=== 测试 ===")
code, out, err = run(
    "source $HOME/.cargo/env && cd /opt/shi/rust && cargo test --release 2>&1 | grep 'test result'",
    timeout=900,
)
print(out)

print("=== 大规模统计 ===")
code, out, err = run("/opt/shi/rust/target/release/stats", timeout=1200)
print(out)

c.close()
