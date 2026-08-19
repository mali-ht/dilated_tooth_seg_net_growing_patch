#!/usr/bin/env python3
"""mesh_viewer_linux.py -- LINUX side of the two-machine mesh pipeline.

An Open3D GUI (Start/Stop Scan buttons, auto-fit camera, true unlit colors) that
connects to mesh_server_win.py over TCP: it sends start/stop commands and renders
the meshes streamed back, in real time. No Frida here.

Run on the Linux GUI PC:
    python3 mesh_viewer_linux.py --host <WINDOWS_SERVER_IP>
    python3 mesh_viewer_linux.py --host 192.168.1.50 --port 8770

Needs: pip install numpy open3d ; and mesh_wire.py in the same folder.
"""

import argparse
import socket
import sys
import threading
import time

import numpy as np
import open3d as o3d

import mesh_wire


def _have_gui() -> bool:
    try:
        _ = o3d.visualization.gui.Application
        _ = o3d.visualization.rendering.Open3DScene
        return True
    except Exception:  # noqa: BLE001
        return False


class Conn:
    """A reconnecting TCP connection to the Windows server. send() and recv()
    are used from different threads; connect() replaces the socket on drop."""

    def __init__(self, host, port, retry=2.0):
        self.host = host
        self.port = port
        self.retry = retry
        self._lock = threading.Lock()
        self.sock = None
        self.on_state = None  # optional callback(str) for status display
        self.connect()

    def _note(self, text):
        print(f"[net] {text}")
        if self.on_state:
            try:
                self.on_state(text)
            except Exception:  # noqa: BLE001
                pass

    def connect(self):
        while True:
            try:
                s = socket.create_connection((self.host, self.port), timeout=5)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self._lock:
                    self.sock = s
                self._note(f"connected to {self.host}:{self.port}")
                return
            except OSError as e:
                self._note(f"connect to {self.host}:{self.port} failed ({e}); retrying in {self.retry:.0f}s")
                time.sleep(self.retry)

    def send(self, header, blob=b""):
        with self._lock:
            s = self.sock
        if s is None:
            return
        try:
            mesh_wire.send_frame(s, header, blob)
        except OSError as e:
            print(f"[net] send failed: {e}", file=sys.stderr)

    def recv(self):
        with self._lock:
            s = self.sock
        if s is None:
            return None
        return mesh_wire.recv_frame(s)  # may raise OSError -> handled by reader


class MeshViewerApp:
    """Open3D GUI window: the live stitched mesh + Start/Stop buttons, auto-fit
    camera, unlit true colors. Start/Stop call send_command(cmd) (-> TCP)."""

    def __init__(self, send_command, title="Runyes live mesh (Linux viewer)", width=1280, height=860):
        self.send_command = send_command
        self.gui = o3d.visualization.gui
        self.rendering = o3d.visualization.rendering

        app = self.gui.Application.instance
        app.initialize()
        self.window = app.create_window(title, width, height)

        self.scene_widget = self.gui.SceneWidget()
        self.scene_widget.scene = self.rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.9, 0.9, 0.9, 1.0])  # light grey
        self.window.add_child(self.scene_widget)

        em = self.window.theme.font_size
        self.panel = self.gui.Vert(0.3 * em, self.gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        self.start_btn = self.gui.Button("Start Scan")
        self.start_btn.set_on_clicked(self._on_start)
        self.stop_btn = self.gui.Button("Stop Scan")
        self.stop_btn.set_on_clicked(self._on_stop)
        self.autofit = self.gui.Checkbox("Auto-fit view")
        self.autofit.checked = True
        self.fitnow_btn = self.gui.Button("Fit now")
        self.fitnow_btn.set_on_clicked(self._on_fit_now)
        self.status = self.gui.Label("connecting…")
        for w in (self.start_btn, self.stop_btn, self.autofit, self.fitnow_btn, self.status):
            self.panel.add_child(w)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)

        self.material = self.rendering.MaterialRecord()
        self.material.shader = "defaultUnlit"           # true per-vertex colors, no shading
        self.material.base_color = [1.0, 1.0, 1.0, 1.0]

        self._region_bounds = {}
        self._fit_bounds = None
        self._camera_ready = False
        _b0 = o3d.geometry.AxisAlignedBoundingBox([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
        self.scene_widget.setup_camera(60.0, _b0, _b0.get_center())

    def _on_layout(self, ctx):
        r = self.window.content_rect
        self.scene_widget.frame = r
        pref = self.panel.calc_preferred_size(ctx, self.gui.Widget.Constraints())
        em = self.window.theme.font_size
        self.panel.frame = self.gui.Rect(r.x + int(0.5 * em), r.y + int(0.5 * em),
                                         max(pref.width, int(16 * em)), max(pref.height, int(14 * em)))

    def _on_start(self):
        self.send_command("start_scan")
        self.status.text = "start requested…"

    def _on_stop(self):
        self.send_command("stop_scan")
        self.status.text = "stop requested…"

    def _on_fit_now(self):
        self._fit_bounds = None
        self._fit_camera(force=True)

    # thread-safe: callable from the network reader thread
    def set_status(self, text):
        self.gui.Application.instance.post_to_main_thread(self.window, lambda: setattr(self.status, "text", text))

    def push(self, mesh):
        self.gui.Application.instance.post_to_main_thread(self.window, lambda: self._apply(mesh))

    def _apply(self, mesh):
        name = f"m_{mesh['modelset']}_{mesh['catalog']}_{mesh['mesh_id']}"
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(mesh["vertices"].astype(np.float64))
        m.triangles = o3d.utility.Vector3iVector(
            mesh["triangles"].astype(np.int32) if mesh["triangles"] is not None else np.empty((0, 3), dtype=np.int32)
        )
        if mesh["normals"] is not None:
            m.vertex_normals = o3d.utility.Vector3dVector(mesh["normals"].astype(np.float64))
        else:
            m.compute_vertex_normals()
        if mesh["colors"] is not None:
            m.vertex_colors = o3d.utility.Vector3dVector(mesh["colors"].astype(np.float64))

        scene = self.scene_widget.scene
        try:
            if scene.has_geometry(name):
                scene.remove_geometry(name)
            scene.add_geometry(name, m, self.material)
        except Exception as e:  # noqa: BLE001
            print(f"[!] viewer add_geometry failed: {e}", file=sys.stderr)
            return

        v = mesh["vertices"]
        if v.size:
            self._region_bounds[name] = (v.min(axis=0), v.max(axis=0))
        self._fit_camera(force=False)

    def _fit_camera(self, force):
        if not self._region_bounds:
            return
        if not force and (not self.autofit.checked) and self._camera_ready:
            return
        mins = np.min([b[0] for b in self._region_bounds.values()], axis=0)
        maxs = np.max([b[1] for b in self._region_bounds.values()], axis=0)
        if not force and self._fit_bounds is not None:
            fmin, fmax = self._fit_bounds
            if not (np.any(mins < fmin) or np.any(maxs > fmax)):
                return
        center = (mins + maxs) / 2.0
        half = np.maximum((maxs - mins) / 2.0, 1e-3)
        margin = 1.7
        emin, emax = center - half * margin, center + half * margin
        self._fit_bounds = (emin, emax)
        bbox = o3d.geometry.AxisAlignedBoundingBox(emin, emax)
        try:
            self.scene_widget.setup_camera(60.0, bbox, bbox.get_center())
            self._camera_ready = True
        except Exception as e:  # noqa: BLE001
            print(f"[!] camera fit failed: {e}", file=sys.stderr)

    def run(self):
        self.gui.Application.instance.run()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="192.168.1.30", help="IP address of the Windows mesh server (mesh_server_win.py) [default: 192.168.1.30]")
    ap.add_argument("--port", type=int, default=mesh_wire.DEFAULT_PORT, help=f"server TCP port (default {mesh_wire.DEFAULT_PORT})")
    ap.add_argument("--retry", type=float, default=2.0, help="seconds between reconnect attempts")
    args = ap.parse_args()

    if not _have_gui():
        print("[!] this Open3D build lacks the GUI framework -- pip install --upgrade open3d", file=sys.stderr)
        return

    conn = Conn(args.host, args.port, args.retry)
    viewer = MeshViewerApp(lambda cmd: conn.send({"type": "cmd", "cmd": cmd}, b""))
    conn.on_state = viewer.set_status  # show connect/disconnect in the panel

    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                frame = conn.recv()
            except OSError:
                frame = None
            if frame is None:
                if stop.is_set():
                    return
                viewer.set_status("server closed -- reconnecting…")
                conn.connect()
                continue
            header, blob = frame
            t = header.get("type")
            if t == "mesh":
                try:
                    viewer.push(mesh_wire.parse_mesh_payload(header, blob))
                except Exception as e:  # noqa: BLE001
                    print(f"[!] parse mesh failed: {e}", file=sys.stderr)
            elif t == "scan_control":
                if header.get("ok"):
                    viewer.set_status(f"scan {header.get('action')}")
                else:
                    viewer.set_status(f"scan control failed: {header.get('error')}")
            elif t == "status":
                viewer.set_status(f"server: module_base {header.get('module_base')}")
            elif t == "error":
                print(f"[!] server-side error: {header}", file=sys.stderr)

    threading.Thread(target=reader, daemon=True).start()
    print(f"[*] viewer up. Click 'Start Scan' to begin. (server {args.host}:{args.port})")
    try:
        viewer.run()  # blocks until the window is closed
    finally:
        stop.set()
        with conn._lock:
            s = conn.sock
        if s is not None:
            try:
                s.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
