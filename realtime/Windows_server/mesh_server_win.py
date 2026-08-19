#!/usr/bin/env python3
"""mesh_server_win.py -- WINDOWS side of the two-machine mesh pipeline.

Wraps IOSAlgo.dll via Frida (loads mesh_intercept.js into 3ds.exe), and bridges
it to a single TCP client (the Linux viewer):
  * every intercepted mesh is relayed to the client as RAW BYTES (no numpy here
    -- this side just moves bytes, so it stays near-zero-latency), and
  * start_scan / stop_scan commands received from the client are forwarded to
    the Frida RPC.

Runs on the scanner PC, in an ADMIN terminal (Frida needs admin to attach to
3ds.exe under Program Files). Needs: pip install frida-tools ; and
mesh_intercept.js + mesh_wire.py in the same folder. Does NOT need numpy/open3d.

    python mesh_server_win.py                 # listen on 0.0.0.0:8770, idle-wait for 3ds.exe
    python mesh_server_win.py --port 9000
"""

import argparse
import socket
import sys
import threading
import time

try:
    import frida
except ImportError:
    print("This needs the 'frida' package: pip install frida-tools", file=sys.stderr)
    raise

import mesh_wire

DEFAULT_PROCESS_NAME = "3ds.exe"
DEFAULT_SCRIPT_PATH = "mesh_intercept.js"

# frida's Python Session has no enumerate_modules(); enumerate in-process via JS.
_MODCHECK_JS = (
    "rpc.exports.has = function (name) {"
    "  name = name.toLowerCase();"
    "  return Process.enumerateModules().some(function (m) {"
    "    return m.name.toLowerCase() === name; }); };"
)


def _find_pids(device, name):
    nl = name.lower()
    return [p.pid for p in device.enumerate_processes() if p.name.lower() == nl]


def _module_loaded(session, module_name):
    try:
        sc = session.create_script(_MODCHECK_JS)
        sc.load()
        try:
            return bool(sc.exports_sync.has(module_name))
        finally:
            try:
                sc.unload()
            except Exception:
                pass
    except Exception:
        return False


def wait_for_target(device, process_name, module_name, poll_interval, timeout):
    """Idle until process_name exists AND has module_name loaded; return
    (session, pid). Skips the short-lived bootstrapper 3ds.exe that exits."""
    start = time.time()
    announced = False
    while True:
        for pid in _find_pids(device, process_name):
            try:
                session = device.attach(pid)
            except Exception:
                continue
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if _module_loaded(session, module_name):
                    print(f"[*] {process_name} (pid {pid}) is up and {module_name} is loaded.")
                    return session, pid
                time.sleep(0.25)
            try:
                session.detach()
            except Exception:
                pass
        if not announced:
            print(f"[*] idle -- waiting for '{process_name}' with '{module_name}' loaded. "
                  "Launch the scanner whenever; Ctrl+C to cancel.")
            announced = True
        if timeout is not None and time.time() - start > timeout:
            raise TimeoutError(f"timed out after {timeout:.0f}s waiting for {process_name} with {module_name}")
        time.sleep(poll_interval)


class ClientLink:
    """Holds the current client socket, guarded so the Frida callback thread and
    the accept thread can share it safely. One client at a time; a new
    connection replaces (and closes) the old."""

    def __init__(self):
        self._lock = threading.Lock()
        self.sock = None

    def set(self, sock):
        with self._lock:
            old = self.sock
            self.sock = sock
        if old is not None and old is not sock:
            try:
                old.close()
            except Exception:
                pass

    def clear(self, sock):
        with self._lock:
            if self.sock is sock:
                self.sock = None

    def send(self, header, blob=b"") -> bool:
        with self._lock:
            s = self.sock
        if s is None:
            return False
        try:
            mesh_wire.send_frame(s, header, blob)
            return True
        except OSError:
            self.clear(s)
            return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0 = all interfaces)")
    ap.add_argument("--port", type=int, default=mesh_wire.DEFAULT_PORT, help=f"TCP port (default {mesh_wire.DEFAULT_PORT})")
    ap.add_argument("--process-name", default=DEFAULT_PROCESS_NAME, help="process to attach to (default 3ds.exe)")
    ap.add_argument("--pid", type=int, default=None, help="attach to this PID instead of by name")
    ap.add_argument("--script", default=DEFAULT_SCRIPT_PATH, help="path to mesh_intercept.js")
    ap.add_argument("--no-wait", action="store_true", help="attach immediately instead of idle-waiting")
    ap.add_argument("--module", default="IOSAlgo.dll", help="module that must be loaded before hooking")
    ap.add_argument("--poll-interval", type=float, default=1.0)
    ap.add_argument("--wait-timeout", type=float, default=None)
    args = ap.parse_args()

    device = frida.get_local_device()
    if args.pid is not None:
        session = device.attach(args.pid)
        desc = f"pid {args.pid}"
    elif args.no_wait:
        session = device.attach(args.process_name)
        desc = args.process_name
    else:
        try:
            session, pid = wait_for_target(device, args.process_name, args.module, args.poll_interval, args.wait_timeout)
        except TimeoutError as e:
            print(f"[!] {e}", file=sys.stderr)
            return
        desc = f"{args.process_name} (pid {pid})"

    with open(args.script, "r", encoding="utf-8") as fh:
        source = fh.read()
    script = session.create_script(source)

    link = ClientLink()

    def on_message(message, data):
        if message.get("type") == "send":
            payload = message.get("payload") or {}
            ptype = payload.get("type")
            if ptype == "mesh":
                link.send(payload, data or b"")            # header + raw geometry blob
            elif ptype in ("scan_control", "status", "error"):
                link.send(payload, b"")                     # status/control (no blob)
        elif message.get("type") == "error":
            print(f"[!] Frida error: {message.get('description')}", file=sys.stderr)

    script.on("message", on_message)
    script.load()
    print(f"[*] attached to {desc}; Frida script loaded.")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[*] listening on {args.host}:{args.port} -- waiting for the Linux viewer to connect.")

    def handle_client(conn, addr):
        print(f"[*] viewer connected from {addr[0]}:{addr[1]}")
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        link.set(conn)
        try:
            while True:
                frame = mesh_wire.recv_frame(conn)
                if frame is None:
                    break
                header, _ = frame
                cmd = header.get("cmd")
                if cmd == "start_scan":
                    try:
                        script.exports_sync.start_scan()
                        print("[*] -> start_scan")
                    except Exception as e:
                        print(f"[!] start_scan failed: {e}", file=sys.stderr)
                elif cmd == "stop_scan":
                    try:
                        script.exports_sync.stop_scan()
                        print("[*] -> stop_scan")
                    except Exception as e:
                        print(f"[!] stop_scan failed: {e}", file=sys.stderr)
        except OSError:
            pass
        finally:
            print("[*] viewer disconnected.")
            link.clear(conn)
            try:
                conn.close()
            except Exception:
                pass

    try:
        while True:
            conn, addr = srv.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            srv.close()
        except Exception:
            pass
        try:
            session.detach()
        except Exception:
            pass


if __name__ == "__main__":
    main()
