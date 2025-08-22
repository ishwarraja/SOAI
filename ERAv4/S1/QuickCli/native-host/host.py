#!/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
import sys
import json
import struct
import subprocess
import datetime
import traceback
import os

LOG_FILE = "/tmp/quickcli_host.log"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.datetime.now()}] {msg}\n")

def read_message():
    raw_length = sys.stdin.buffer.read(4)
    if not raw_length:
        return None
    message_length = struct.unpack("=I", raw_length)[0]
    message = sys.stdin.buffer.read(message_length).decode("utf-8")
    return json.loads(message)

def send_message(message):
    encoded = json.dumps(message).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("=I", len(encoded)))
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()

def main():
    log("host.py started")
    current_dir = os.path.expanduser("~")  # start in home directory

    try:
        while True:
            msg = read_message()
            if msg is None:
                log("No message received, exiting")
                break

            cmd = msg.get("command", "").strip()
            log(f"Received: {cmd} (cwd={current_dir})")

            if not cmd:
                send_message({"ok": False, "msg": "Empty command"})
                continue

            # Special case: handle "cd"
            if cmd.startswith("cd"):
                parts = cmd.split(maxsplit=1)
                if len(parts) == 1 or parts[1].strip() in ("~", ""):
                    new_dir = os.path.expanduser("~")
                else:
                    target = parts[1].strip()
                    if os.path.isabs(target):
                        new_dir = target
                    else:
                        new_dir = os.path.join(current_dir, target)

                if os.path.isdir(new_dir):
                    current_dir = os.path.abspath(new_dir)
                    send_message({"ok": True, "msg": f"Changed directory to {current_dir}"})
                    log(f"Changed directory to {current_dir}")
                else:
                    send_message({"ok": False, "msg": f"No such directory: {new_dir}"})
                continue


            # Execute other commands in the current_dir
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=10, cwd=current_dir
                )
                output = result.stdout.strip() or result.stderr.strip() or "(no output)"
                send_message({"ok": result.returncode == 0, "msg": output})
                log(f"Sent response: {output}")
            except Exception as e:
                log(f"Execution error: {e}")
                send_message({"ok": False, "msg": f"Execution error: {e}"})
    except Exception as e:
        log("Fatal error: " + str(e))
        log(traceback.format_exc())
        send_message({"ok": False, "msg": f"Fatal error: {e}"})

if __name__ == "__main__":
    main()
