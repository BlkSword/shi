"""部署 Rust 二进制到测试机并运行。"""
import os
import sys
import paramiko

HOST = os.environ.get("SHI_HOST", "103.242.14.227")
USER = os.environ.get("SHI_USER", "root")
PASS = os.environ.get("SHI_PASS", "")

def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=20, banner_timeout=20)
    return c

def run(c, cmd, timeout=600):
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err

def main():
    c = connect()

    # 1. 检查环境
    code, out, err = run(c, "uname -m; which cargo || echo NO_CARGO; which gcc || echo NO_GCC")
    print("=== 远程环境 ===")
    print(out)

    # 2. 上传源码
    sftp = c.open_sftp()
    print("=== 上传源码 ===")

    import tarfile
    import io
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add("rust", arcname="rust")
    buf.seek(0)
    sftp.putfo(buf, "/tmp/shi-rust.tar.gz")
    print("  uploaded", buf.tell(), "bytes")

    code, out, err = run(c, "mkdir -p /opt/shi && cd /opt/shi && tar xzf /tmp/shi-rust.tar.gz && ls rust/")
    print(out)
    if err:
        print("STDERR:", err)

    sftp.close()
    c.close()

if __name__ == "__main__":
    main()
