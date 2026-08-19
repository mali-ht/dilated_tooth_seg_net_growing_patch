#!/usr/bin/env python3
"""mesh_relay.py -- local fan-out relay for the mesh link, so the raw-color viewer
(mesh_viewer_linux.py, unmodified) and the new segmentation viewer
(mesh_viewer_segmented.py) can both watch the same live scan without either of
them needing a second, independent connection to the Windows mesh server.

Whether mesh_server_win.py actually supports multiple simultaneous TCP clients
isn't verifiable without its source, so this makes exactly ONE real connection
upstream and fans frames out locally instead - a relay is safe regardless of
that answer, a second direct connection to the real server might not be.

    Windows mesh_server_win.py
              |  (one real TCP connection)
              v
      mesh_relay.py  (this file, runs on the Linux box)
         /         \\
        v           v
 mesh_viewer_linux.py   mesh_viewer_segmented.py
 (--host 127.0.0.1      (--host 127.0.0.1
  --port <local_port>)   --port <local_port>)

Frames flow both ways: mesh/status frames upstream->relay->every local client;
command frames (start_scan/stop_scan) from ANY local client->relay->upstream
(both viewers are watching the same real scan, so either one's Start/Stop
button controls it - that's the intended behavior, not a bug).

Run on the Linux box, in place of pointing the viewers directly at Windows:
    python3 mesh_relay.py --host <WINDOWS_SERVER_IP>
    python3 mesh_viewer_linux.py --host 127.0.0.1 --port 8780
    python3 mesh_viewer_segmented.py --host 127.0.0.1 --port 8780 --ckpt <path>
"""

import argparse
import queue
import socket
import sys
import threading
import time

import mesh_wire
from mesh_viewer_linux import Conn

DEFAULT_LOCAL_PORT = 8780


def _debug(msg):
    print(f"[debug:relay] {msg}")


class DownstreamClient:
    """One locally-connected viewer (raw or segmented). Frames meant for it are
    pushed onto its own queue and sent by its own thread, so one slow/stuck
    local client can't stall delivery to the others."""

    def __init__(self, sock, addr, on_command):
        self.sock = sock
        self.addr = addr
        self.on_command = on_command
        self.out_q = queue.Queue(maxsize=64)
        self.alive = True
        self.n_pushed = 0
        self.n_dropped = 0
        threading.Thread(target=self._sender, daemon=True).start()
        threading.Thread(target=self._reader, daemon=True).start()

    def push(self, header, blob):
        try:
            self.out_q.put_nowait((header, blob))
            self.n_pushed += 1
        except queue.Full:
            self.n_dropped += 1
            print(f"[relay] downstream {self.addr} too slow, dropping a frame "
                  f"(pushed={self.n_pushed} dropped={self.n_dropped} total)", file=sys.stderr)

    def _sender(self):
        while self.alive:
            item = self.out_q.get()
            if item is None:
                return
            header, blob = item
            try:
                mesh_wire.send_frame(self.sock, header, blob)
                _debug(f"-> {self.addr}: type={header.get('type')} "
                       f"mesh_id={header.get('mesh_id')} bytes={len(blob)}")
            except OSError as e:
                _debug(f"send to {self.addr} failed: {e}")
                self.close()
                return

    def _reader(self):
        # anything a local viewer sends us (Start/Stop commands) goes straight upstream
        while self.alive:
            try:
                frame = mesh_wire.recv_frame(self.sock)
            except OSError:
                frame = None
            if frame is None:
                self.close()
                return
            header, blob = frame
            _debug(f"<- {self.addr}: {header}")
            self.on_command(header, blob)

    def close(self):
        if not self.alive:
            return
        self.alive = False
        self.out_q.put_nowait(None)
        try:
            self.sock.close()
        except OSError:
            pass
        print(f"[relay] downstream {self.addr} disconnected (pushed={self.n_pushed} dropped={self.n_dropped})")


class Relay:
    def __init__(self, upstream_host, upstream_port, local_port, retry):
        self.upstream = Conn(upstream_host, upstream_port, retry)  # blocks until connected
        self.local_port = local_port
        self._clients = []
        self._clients_lock = threading.Lock()
        # last known mesh frame per mesh_id, replayed to a newly-connected client so it isn't
        # missing everything scanned before it joined - a raw viewer started first, scanned for a
        # while, THEN the segmented viewer started, would otherwise never see the earlier chunks
        # at all (only future updates to them)
        self._latest_by_mesh_id = {}
        self._latest_lock = threading.Lock()
        self._n_mesh_frames = 0
        self._n_bytes = 0
        self._t_start = time.time()

    def _on_downstream_command(self, header, blob):
        _debug(f"forwarding command upstream: {header}")
        self.upstream.send(header, blob)

    def _accept_loop(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", self.local_port))
        srv.listen(4)
        print(f"[relay] listening for local viewers on 0.0.0.0:{self.local_port}")
        while True:
            sock, addr = srv.accept()
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[relay] downstream connected: {addr}")
            client = DownstreamClient(sock, addr, self._on_downstream_command)
            with self._latest_lock:
                catch_up = list(self._latest_by_mesh_id.values())
            for header, blob in catch_up:
                client.push(header, blob)
            if catch_up:
                print(f"[relay] caught up {addr} with {len(catch_up)} known mesh_id(s)")
            with self._clients_lock:
                self._clients = [c for c in self._clients if c.alive] + [client]

    def _upstream_reader_loop(self):
        while True:
            try:
                frame = self.upstream.recv()
            except OSError:
                frame = None
            if frame is None:
                print("[relay] upstream closed - reconnecting…")
                self.upstream.connect()
                continue
            header, blob = frame
            if header.get("type") == "mesh":
                with self._latest_lock:
                    self._latest_by_mesh_id[header.get("mesh_id")] = (header, blob)
                self._n_mesh_frames += 1
                self._n_bytes += len(blob)
                if self._n_mesh_frames % 20 == 0:
                    elapsed = time.time() - self._t_start
                    _debug(f"throughput: {self._n_mesh_frames} mesh frames, "
                           f"{self._n_bytes / 1e6:.1f}MB total, "
                           f"{self._n_mesh_frames / elapsed:.1f} frames/s avg, "
                           f"{len(self._latest_by_mesh_id)} distinct mesh_id(s) known")
            else:
                _debug(f"upstream frame: {header}")
            with self._clients_lock:
                clients = [c for c in self._clients if c.alive]
            for c in clients:
                c.push(header, blob)

    def run(self):
        threading.Thread(target=self._accept_loop, daemon=True).start()
        self._upstream_reader_loop()  # blocks forever


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="192.168.1.30", help="Windows mesh server IP")
    ap.add_argument("--port", type=int, default=mesh_wire.DEFAULT_PORT, help="Windows mesh server port")
    ap.add_argument("--local_port", type=int, default=DEFAULT_LOCAL_PORT,
                     help="port local viewers (mesh_viewer_linux.py, mesh_viewer_segmented.py) connect to")
    ap.add_argument("--retry", type=float, default=2.0)
    args = ap.parse_args()

    relay = Relay(args.host, args.port, args.local_port, args.retry)
    relay.run()


if __name__ == "__main__":
    main()
