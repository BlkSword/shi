"""SSH 远程执行工具（密码从环境变量读取，不落盘）。"""
import os
import sys
import paramiko

HOST = os.environ.get("SHI_HOST", "103.242.14.227")
USER = os.environ.get("SHI_USER", "root")
PASS = os.environ.get("SHI_PASS", "")

def run(cmd, timeout=120):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=20, banner_timeout=20)
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    c.close()
    return code, out, err

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "echo hello"
    code, out, err = run(cmd)
    print(out)
    if err:
        print("STDERR:", err, file=sys.stderr)
    sys.exit(code)
