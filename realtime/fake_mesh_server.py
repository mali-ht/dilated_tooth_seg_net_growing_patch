#!/usr/bin/env python3
"""fake_mesh_server.py -- stands in for the real mesh_server_win.py + scanner
hardware, so mesh_relay.py and mesh_viewer_segmented.py can be tested end to
end without either. NOT part of the real two-machine pipeline - test only,
used because this session has no access to the real Windows server/scanner.

Replays a real cached Teeth3DS arch as several independently-growing spatial
"regions" (simulating mesh_id-keyed reconstruction blocks a real scanner would
produce), each region's snapshot re-sent - larger each time - on its own
schedule after a start_scan command, over the exact mesh_wire protocol used
by the real server - so anything downstream sees traffic indistinguishable in
shape from the real thing: multiple mesh_ids, repeated updates to the SAME
mesh_id (growing), a whole that only becomes coherent once merged.

Run:
    python3 fake_mesh_server.py --pt_path <path to a processed_w5 cache file> --port 8770
"""

import argparse
import os
import socket
import sys
import threading
import time

import numpy as np

import mesh_wire

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root -
# this file moved from testing/realtime/ to realtime/ (one level shallower), so this now only
# needs to strip 2 path components (file -> realtime -> root), not 3
from dataset.patch_generator import load_cached_arch  # noqa: E402


def _build_regions(mesh, n_regions=6, seed=0):
    """Bin face indices into n_regions groups by centroid position along the arch's own
    longest axis (via PCA), roughly simulating spatially-distinct reconstruction blocks."""
    centroids = mesh.triangles_center
    centered = centroids - centroids.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis_pos = centered @ vt[0]
    order = np.argsort(axis_pos)
    return np.array_split(order, n_regions)


def _chunk_bytes(mesh, face_idx):
    """One mesh_wire chunk (locally-indexed vertices/triangles/normals) for face_idx."""
    faces = mesh.faces[face_idx]
    used_v = np.unique(faces)
    remap = -np.ones(len(mesh.vertices), dtype=np.int64)
    remap[used_v] = np.arange(len(used_v))
    vertices = mesh.vertices[used_v].astype("<f4")
    normals = mesh.vertex_normals[used_v].astype("<f4")
    triangles = remap[faces].astype("<u4")
    return vertices, normals, triangles


def _send_mesh_frame(sock, mesh_id, vertices, normals, triangles, seq):
    v_bytes, n_bytes, t_bytes = vertices.tobytes(), normals.tobytes(), triangles.tobytes()
    header = {
        "type": "mesh", "mesh_id": mesh_id, "modelset": "fake", "catalog": "fake",
        "vertex_count": len(vertices), "face_count": len(triangles),
        "layout": {"vertices_bytes": len(v_bytes), "normals_bytes": len(n_bytes),
                   "colors_bytes": 0, "triangles_bytes": len(t_bytes)},
        "seq": seq, "source": "fake_mesh_server", "ts": time.time(),
    }
    mesh_wire.send_frame(sock, header, v_bytes + n_bytes + t_bytes)


class ScanSession:
    """One simulated scan for one connected client: grows each region's revealed face
    count in small steps, re-sending that region's mesh_id whenever it changes, until
    stopped or the client disconnects."""

    def __init__(self, sock, mesh, regions, tick_seconds, growth_per_tick):
        self.sock = sock
        self.mesh = mesh
        self.regions = regions
        self.tick_seconds = tick_seconds
        self.growth_per_tick = growth_per_tick
        self._stop = threading.Event()
        self._thread = None
        self._seq = 0

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        revealed = np.zeros(len(self.regions))
        active_region = 0
        while not self._stop.is_set() and active_region < len(self.regions):
            revealed[active_region] = min(1.0, revealed[active_region] + self.growth_per_tick)
            region = self.regions[active_region]
            n_faces = max(1, int(round(len(region) * revealed[active_region])))
            vertices, normals, triangles = _chunk_bytes(self.mesh, region[:n_faces])
            try:
                _send_mesh_frame(self.sock, active_region, vertices, normals, triangles, self._seq)
            except OSError:
                return
            self._seq += 1
            if revealed[active_region] >= 1.0:
                active_region += 1
            time.sleep(self.tick_seconds)


def _handle_client(sock, addr, mesh, regions, tick_seconds, growth_per_tick):
    print(f"[fake-server] client connected: {addr}")
    session = ScanSession(sock, mesh, regions, tick_seconds, growth_per_tick)
    try:
        while True:
            frame = mesh_wire.recv_frame(sock)
            if frame is None:
                break
            header, _blob = frame
            if header.get("type") == "cmd":
                cmd = header.get("cmd")
                print(f"[fake-server] {addr} -> {cmd}")
                if cmd == "start_scan":
                    session.start()
                    mesh_wire.send_frame(sock, {"type": "scan_control", "action": "start_scan", "ok": True})
                elif cmd == "stop_scan":
                    session.stop()
                    mesh_wire.send_frame(sock, {"type": "scan_control", "action": "stop_scan", "ok": True})
    except OSError:
        pass
    finally:
        session.stop()
        sock.close()
        print(f"[fake-server] client disconnected: {addr}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt_path", required=True, help="A processed_w5 cached arch (data_*.pt)")
    ap.add_argument("--port", type=int, default=mesh_wire.DEFAULT_PORT)
    ap.add_argument("--n_regions", type=int, default=6)
    ap.add_argument("--tick_seconds", type=float, default=0.3)
    ap.add_argument("--growth_per_tick", type=float, default=0.25)
    args = ap.parse_args()

    mesh, _labels = load_cached_arch(args.pt_path)
    regions = _build_regions(mesh, args.n_regions)
    print(f"[fake-server] loaded {args.pt_path}: {len(mesh.faces)} faces, {len(regions)} simulated regions")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(4)
    print(f"[fake-server] listening on 0.0.0.0:{args.port}")
    while True:
        sock, addr = srv.accept()
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        threading.Thread(target=_handle_client, args=(sock, addr, mesh, regions, args.tick_seconds,
                                                        args.growth_per_tick), daemon=True).start()


if __name__ == "__main__":
    main()
