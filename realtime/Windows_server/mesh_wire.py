"""mesh_wire.py -- shared wire protocol for the mesh link between the Windows
server (mesh_server_win.py) and the Linux viewer (mesh_viewer_linux.py).

ONE framing scheme, both directions:
    [4B big-endian header_len][4B big-endian blob_len][header JSON utf-8][blob]

  * Mesh frames  (server -> viewer): header = the Frida 'mesh' payload
    (type, mesh_id, modelset, catalog, vertex_count, face_count, layout, ...),
    blob = the concatenated binary geometry (vertices|normals|colors|triangles).
  * Command frames (viewer -> server): header = {"type":"cmd","cmd":"start_scan"},
    blob empty.
  * Status frames (server -> viewer): header = a scan_control/status/error dict,
    blob empty.

numpy is imported lazily (only inside parse_mesh_payload) so the Windows server
side -- which only relays raw bytes -- does not need numpy installed.
"""

import json
import struct

DEFAULT_PORT = 8770
_HDR = struct.Struct(">II")  # header_len, blob_len


def send_frame(sock, header: dict, blob: bytes = b"") -> None:
    """Serialize + send one frame. Raises OSError if the socket is broken."""
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    sock.sendall(_HDR.pack(len(hb), len(blob)))
    sock.sendall(hb)
    if blob:
        sock.sendall(blob)


def _recvall(sock, n: int):
    """Read exactly n bytes, or None if the peer closed the connection."""
    parts = []
    got = 0
    while got < n:
        chunk = sock.recv(min(1 << 20, n - got))
        if not chunk:
            return None
        parts.append(chunk)
        got += len(chunk)
    return b"".join(parts)


def recv_frame(sock):
    """Return (header_dict, blob_bytes), or None if the peer closed."""
    head = _recvall(sock, _HDR.size)
    if head is None:
        return None
    hlen, blen = _HDR.unpack(head)
    hb = _recvall(sock, hlen)
    if hb is None:
        return None
    header = json.loads(hb.decode("utf-8"))
    blob = b""
    if blen:
        blob = _recvall(sock, blen)
        if blob is None:
            return None
    return header, blob


def parse_mesh_payload(header: dict, blob: bytes) -> dict:
    """Reassemble a mesh frame into numpy arrays (viewer side only). Buffer
    layout: vertices(3xf32) | normals(3xf32) | colors(3xu8) | triangles(3xu32),
    with byte lengths given in header['layout']."""
    import numpy as np

    layout = header["layout"]
    vc = header["vertex_count"]
    fc = header["face_count"]

    off = 0
    vertices = np.frombuffer(blob, "<f4", vc * 3, off).reshape(-1, 3)
    off += layout["vertices_bytes"]

    normals = None
    if layout["normals_bytes"]:
        normals = np.frombuffer(blob, "<f4", vc * 3, off).reshape(-1, 3)
    off += layout["normals_bytes"]

    colors = None
    if layout["colors_bytes"]:
        # on-wire uint8 RGB888 (0-255) -> float 0-1 for Open3D
        colors = np.frombuffer(blob, np.uint8, vc * 3, off).reshape(-1, 3).astype(np.float64) / 255.0
    off += layout["colors_bytes"]

    triangles = None
    if layout["triangles_bytes"]:
        triangles = np.frombuffer(blob, "<u4", fc * 3, off).reshape(-1, 3)

    return {
        "mesh_id": header.get("mesh_id"),
        "modelset": header.get("modelset"),
        "catalog": header.get("catalog"),
        "vertex_count": vc,
        "face_count": fc,
        "vertices": vertices,
        "normals": normals,
        "colors": colors,
        "triangles": triangles,
        "layout": layout,
        "seq": header.get("seq"),
        "source": header.get("source"),
        "ts": header.get("ts"),
    }
