/*
 * mesh_intercept.js -- intercept ACCEPTED (post-fusion, published-for-display)
 * meshes live out of IOSAlgo.dll's real-time stitching pipeline, from inside
 * the vendor's own process, during an active scan.
 *
 * WHY THIS HOOK POINT
 *   14_RECONSTRUCTION_STITCHING_PIPELINE.md (Stage 3, "OnlineMergingManager")
 *   documents a producer/consumer real-time architecture:
 *     ThreadMG (producer) -> TFOnlineMeshIntegrated::ExtractAndPublishMesh
 *       (runs volumetric fusion, extracts a new triangle mesh, fixes
 *       coordinate handedness + winding, stores a versioned snapshot)
 *     -> signals a condvar -> ThreadMP (consumer) wakes, pops the snapshot,
 *       calls FUN_1800bf820 to hand it off to "whatever holds the current
 *       displayable mesh state" -- i.e. the exact moment a newly-fused mesh
 *       is accepted and made available to a live viewer.
 *   FUN_1800bf820 is exactly that hand-off: the function registry names it
 *   OnlineMergingManager::ThreadMP_Publish (confirmed via literal
 *   std::_Ref_count_obj2<OutDataMeshAdd/OutDataAuxMeshAdd/OutDataAuxMeshRemove>
 *   vftable installs inside its own body -- read directly from
 *   decompiled_all/IOSAlgo/1800bf820_FUN_1800bf820.c this session). Hooking
 *   its ENTRY gives you every mesh update right as it's accepted for
 *   display, already sign/winding-corrected by ExtractAndPublishMesh
 *   upstream -- this is "the accepted mesh during scanning" the way a live
 *   viewer would see it, not a raw per-view fragment.
 *
 *   Two independent extraction paths were found by reading FUN_1800bf820's
 *   decompiled body directly (both confirmed via its own use of the real,
 *   named `aMesh::GetMeshID` accessor as a validity gate on the exact
 *   pointers extracted below -- not guessed):
 *
 *   PATH A -- "aux mesh add" (single mesh), taken when *(char*)(*param_2+0x60) != 0:
 *     aMesh* = *(aMesh**)(*param_2 + 0x50)                      -- direct pointer
 *
 *   PATH B -- "bulk mesh add" (batch of per-region updates), taken when
 *   *(char*)(*param_2+0x60) == 0: *param_2 + 0x10 holds a pointer to a
 *   std::vector<shared_ptr<OutDataMeshAdd>>-shaped object (begin/end pointer
 *   pair at +0x00/+0x08 of the pointee); each element is a 16-byte
 *   {T*, control_block*} pair (MSVC shared_ptr layout). T* (element+0x00) IS
 *   the OutDataMeshAdd payload, which IS the aMesh (meshID/modelset/catalog at
 *   +0x00/+0x04/+0x08) with the Tfdt::Mesh embedded INLINE at +0x10:
 *     aMesh = *(T**)(vectorElement + 0x00)   -- the payload itself
 *     Tfdt  = aMesh + 0x10                    -- embedded by value, NOT a deref
 *   (The original file mislabeled aMesh as payload+0x10 and then dereferenced
 *   +0x10 again as a Tfdt pointer -- the bug that produced zero meshes, fixed
 *   2026-08-07 and confirmed against a live capture.)
 *
 * aMesh / Tfdt::Mesh STRUCT LAYOUT
 *   Full field table in Investigation/Documentation/11_AMESH_STRUCT_LAYOUT.md,
 *   originally recovered from model_base_tool.exe (which retains full C++
 *   symbols). That document's own "caveat" section flags this layout as
 *   high-confidence-but-not-independently-verified *inside* IOSAlgo.dll
 *   specifically. This session closes that gap: IOSAlgo.dll's own
 *   Tfdt::Mesh::SetVertices/SetVerticesUV/SetVerticesQualities/
 *   SetVerticesAttributes (real, named, symbol-confirmed functions at RVA
 *   0x446fb0/0x447280/0x4471f0/0x447170 respectively -- see
 *   iosalgo_function_registry.tsv) write to exactly the offsets
 *   (+0x08/+0x20/+0x30/+0x38) this layout predicts. The layout below is
 *   therefore confirmed directly inside IOSAlgo.dll, not just inherited.
 *
 *   aMesh (the OutDataMeshAdd payload / vector-element T* in the bulk path):
 *     +0x00 int meshID | +0x04 int modelset | +0x08 int catalog
 *     +0x10 Tfdt::Mesh  -- EMBEDDED BY VALUE here (live-confirmed 2026-08-07),
 *                          NOT a pointer. Read the Tfdt fields below at
 *                          aMesh+0x10 + their offset.
 *   Tfdt::Mesh:
 *     +0x00 int vertexCount        +0x04 int faceCount
 *     +0x08 float* vertices (3f/vertex, XYZ)
 *     +0x10 float* normals  (3f/vertex, XYZ)
 *     +0x18 uchar* colors   (3 BYTES/vertex, RGB888 0-255 -- NOT floats; see
 *                            SetVerticesColors@180446b40. model_base_tool.exe
 *                            treats these as floats, but IOSAlgo.dll does not.)
 *     +0x20 float* vertexUV (2f/vertex)
 *     +0x28 float* triangleUV (6f/face, per-corner)
 *     +0x30 float* vertexQualities (1f/vertex)
 *     +0x38 ushort* vertexAttributes (bitflags/vertex)
 *     +0x40 float* triangleQualities (1f/face)
 *     +0x48 ushort* triangleAttributes (bitflags/face)
 *     +0x50 uint* triangles (3 indices/face)
 *     +0x58 std::string textureFileName (not extracted here)
 *
 * WHAT THIS SCRIPT DOES NOT DO (by design, per project scope)
 *   No TCP/IP, no visualization. Every extracted mesh is handed to the
 *   Python side (mesh_intercept.py) via Frida's send()/on('message') RPC
 *   as one JSON metadata message + one concatenated raw binary buffer
 *   (vertices, then normals, then colors, then triangles, back-to-back --
 *   lengths given in the JSON `layout` field). What you do with it there
 *   (forward over your own socket, save, etc.) is up to you.
 *
 * CONFIDENCE
 *   PATH A / PATH B extraction and the aMesh/Tfdt::Mesh offsets: high --
 *   read directly from decompiled source this session, cross-confirmed by
 *   two independent Set* function addresses inside the same DLL.
 *   Not yet validated against a live capture (this script is new). Run it
 *   during a real scan first and sanity-check vertex_count/face_count and
 *   a few raw vertex values (should be small mm-scale floats, not huge/NaN)
 *   before trusting the stream -- the project's established methodology
 *   throughout this investigation (see ios_algo_hooks.js's own header) has
 *   repeatedly found "looks structurally right" and "live-verified" to be
 *   different things.
 *
 * USAGE
 *   Run ON the Windows machine, alongside 3ds.exe (this hooks the vendor
 *   process locally -- see mesh_intercept.py's own header for the split
 *   between this machine and the separate Linux robot-stack machine).
 *     frida -p <pid> -l mesh_intercept.js          (manual/CLI use)
 *   or driven from Python (recommended, see mesh_intercept.py):
 *     python mesh_intercept.py
 *
 * IMAGE BASE / RVA
 *   Same convention as ios_algo_hooks.js: IOSAlgo.dll's confirmed preferred
 *   ImageBase is 0x180000000 (pefile + Ghidra getImageBase(), confirmed
 *   twice independently in that script's own header). RVA = VA - 0x180000000.
 *   This script resolves the module's ACTUAL runtime base via
 *   Process.findModuleByName() and adds RVAs to that, so it's correct even
 *   if ASLR relocated the DLL.
 */

'use strict';

const MODULE_NAME = 'IOSAlgo.dll';

// Set true to also hook OnlineMergingManager::AWM (RVA 0xb8520), the
// ingestion entry point upstream of fusion -- gives per-view/per-submission
// meshes BEFORE stitching, not the accepted/fused result. Off by default:
// the pointer semantics of its 5th argument (args[4]) are inferred (a
// pointer to a 16-byte shared_ptr-shaped temp whose first qword is the
// real mesh pointer, per the "mesh=%p" log format string using
// `*(undefined8*)param_5` directly) but not independently re-derived with
// the same rigor as PATH A/B above. Kept for anyone who wants pre-fusion
// per-view data too.
const ENABLE_AWM_INGEST_HOOK = false;

// ---------------------------------------------------------------------
// aMesh / Tfdt::Mesh struct offsets (see header)
// ---------------------------------------------------------------------
const AMESH_MESH_ID = 0x00;
const AMESH_MODELSET = 0x04;
const AMESH_CATALOG = 0x08;
const AMESH_TFDT_PTR = 0x10;

const TFDT_VERTEX_COUNT = 0x00;
const TFDT_FACE_COUNT = 0x04;
const TFDT_VERTICES = 0x08;
const TFDT_NORMALS = 0x10;
const TFDT_COLORS = 0x18;
const TFDT_TRIANGLES = 0x50;

const MAX_VERTS = 3000000; // generous sanity ceiling, real scans are far smaller
const MAX_FACES = 6000000;

let meshSeq = 0;

// ---------------------------------------------------------------------
// Core extractor -- given a raw aMesh* believed valid, validates it and,
// if it passes, reads its geometry and ships it to Python. Returns true
// if a mesh was sent.
// ---------------------------------------------------------------------
function extractAndSendAMesh(aMeshPtr, sourceTag) {
  if (aMeshPtr === null || aMeshPtr.isNull()) return false;

  let meshId, modelset, catalog, tfdtPtr;
  try {
    meshId = aMeshPtr.add(AMESH_MESH_ID).readS32();
    modelset = aMeshPtr.add(AMESH_MODELSET).readS32();
    catalog = aMeshPtr.add(AMESH_CATALOG).readS32();
    // FIX (live-confirmed 2026-08-07): the Tfdt::Mesh is embedded BY VALUE at
    // aMesh+0x10 -- it is NOT a pointer. The previous code called .readPointer()
    // here, one indirection too many: it read the normals FLOAT pointer as if it
    // were the Tfdt base, producing garbage vertexCounts and ZERO extracted
    // meshes. Verified against decompiled GetMeshID/SetVertices@180446fb0 and a
    // live hexdump: vertexCount/faceCount/buffer pointers sit directly at
    // aMesh+0x10. The meshID<=0 gate was also dropped -- the payload's +0x00 is
    // a region id (0 is valid); vertexCount below is the real validity gate,
    // which also rejects the vendor's 999999 sentinel / empty aux records.
    tfdtPtr = aMeshPtr.add(AMESH_TFDT_PTR); // inline struct base -- do NOT deref
  } catch (e) {
    return false; // aMeshPtr wasn't valid memory -- wrong source path, not a crash
  }

  let vertexCount, faceCount, verticesPtr, normalsPtr, colorsPtr, trianglesPtr;
  try {
    vertexCount = tfdtPtr.add(TFDT_VERTEX_COUNT).readS32();
    faceCount = tfdtPtr.add(TFDT_FACE_COUNT).readS32();
    if (vertexCount <= 0 || vertexCount > MAX_VERTS) return false;
    if (faceCount < 0 || faceCount > MAX_FACES) return false;
    verticesPtr = tfdtPtr.add(TFDT_VERTICES).readPointer();
    if (verticesPtr.isNull()) return false;
    normalsPtr = tfdtPtr.add(TFDT_NORMALS).readPointer();
    colorsPtr = tfdtPtr.add(TFDT_COLORS).readPointer();
    trianglesPtr = tfdtPtr.add(TFDT_TRIANGLES).readPointer();
  } catch (e) {
    return false;
  }

  const vertsBytes = vertexCount * 3 * 4;                                  // 3 x float32
  const normalsBytes = (!normalsPtr.isNull()) ? vertexCount * 3 * 4 : 0;   // 3 x float32
  // FIX (2026-08-07): colors are uint8 RGB888 -- 3 BYTES/vertex (0-255), NOT 3
  // floats. Confirmed via SetVerticesColors@180446b40 (memcpy vertexCount*3
  // bytes) and the 3ds live renderer using a Vec4ub color array with no texture.
  // Reading it as 3 float32 (12 bytes/vertex) mis-strided the buffer -> rainbow.
  // The Python side divides these bytes by 255 to get 0-1 RGB for the viewer.
  const colorsBytes = (!colorsPtr.isNull()) ? vertexCount * 3 : 0;         // 3 x uint8
  const triBytes = (faceCount > 0 && !trianglesPtr.isNull()) ? faceCount * 3 * 4 : 0;
  const total = vertsBytes + normalsBytes + colorsBytes + triBytes;

  let combined;
  try {
    combined = new Uint8Array(total);
    let off = 0;
    combined.set(new Uint8Array(verticesPtr.readByteArray(vertsBytes)), off);
    off += vertsBytes;
    if (normalsBytes) {
      combined.set(new Uint8Array(normalsPtr.readByteArray(normalsBytes)), off);
    }
    off += normalsBytes;
    if (colorsBytes) {
      combined.set(new Uint8Array(colorsPtr.readByteArray(colorsBytes)), off);
    }
    off += colorsBytes;
    if (triBytes) {
      combined.set(new Uint8Array(trianglesPtr.readByteArray(triBytes)), off);
    }
  } catch (e) {
    send({ type: 'error', where: 'extractAndSendAMesh/read', source: sourceTag, error: e.toString() });
    return false;
  }

  meshSeq++;
  send({
    type: 'mesh',
    seq: meshSeq,
    source: sourceTag,
    mesh_id: meshId,
    modelset: modelset,
    catalog: catalog,
    vertex_count: vertexCount,
    face_count: faceCount,
    layout: {
      vertices_bytes: vertsBytes,
      normals_bytes: normalsBytes,
      colors_bytes: colorsBytes,
      triangles_bytes: triBytes,
    },
    ts: Date.now(),
  }, combined.buffer);
  return true;
}

// ---------------------------------------------------------------------
// PRIMARY HOOK: OnlineMergingManager::ThreadMP_Publish (RVA 0xbf820)
// ---------------------------------------------------------------------
function installThreadMPPublishHook(base) {
  const RVA = 0xbf820;
  const addr = base.add(RVA);
  Interceptor.attach(addr, {
    onEnter(args) {
      try {
        const recPtr = args[1].readPointer(); // *param_2
        const flag = recPtr.add(0x60).readU8();

        if (flag === 0) {
          // PATH B: bulk mesh-add, vector<shared_ptr<OutDataMeshAdd>> at *param_2+0x10
          const vecDescPtr = recPtr.add(0x10).readPointer();
          if (!vecDescPtr.isNull()) {
            const beginPtr = vecDescPtr.readPointer();
            const endPtr = vecDescPtr.add(0x08).readPointer();
            let item = beginPtr;
            let guard = 0;
            while (!item.equals(endPtr) && guard < 10000) {
              let itemBasePtr = null;
              try {
                itemBasePtr = item.readPointer();
              } catch (e) { /* stop this element, keep scanning */ }
              if (itemBasePtr !== null && !itemBasePtr.isNull()) {
                // itemBasePtr (the shared_ptr's T*) IS the aMesh payload:
                // meshID/modelset/catalog at +0x00/+0x04/+0x08, Tfdt inline at
                // +0x10. (Old code passed +0x10 here, which mislabeled the Tfdt
                // as the aMesh -- see extractAndSendAMesh for the full fix.)
                extractAndSendAMesh(itemBasePtr, 'ThreadMP_Publish/bulk');
              }
              item = item.add(0x10);
              guard++;
            }
          }
        } else {
          // PATH A: single aux-mesh-add, aMesh* at *param_2+0x50
          const aMeshPtr = recPtr.add(0x50).readPointer();
          extractAndSendAMesh(aMeshPtr, 'ThreadMP_Publish/aux');
        }
      } catch (e) {
        send({ type: 'error', where: 'ThreadMP_Publish hook', error: e.toString() });
      }
    },
  });
  console.log('[*] hooked OnlineMergingManager::ThreadMP_Publish at ' + addr + ' (RVA 0x' + RVA.toString(16) + ')');
}

// ---------------------------------------------------------------------
// OPTIONAL HOOK: OnlineMergingManager::AWM (RVA 0xb8520) -- pre-fusion
// ingestion, off by default, see ENABLE_AWM_INGEST_HOOK above.
// Signature (fully read, high confidence):
//   void AWM(OMM* this, int modelset, int catalog, int type, <mesh handle>)
// ---------------------------------------------------------------------
function installAWMIngestHook(base) {
  const RVA = 0xb8520;
  const addr = base.add(RVA);
  Interceptor.attach(addr, {
    onEnter(args) {
      try {
        const type = args[3].toInt32();
        const meshHandlePtr = args[4]; // pointer to a 16-byte {T*, ctrl*} temp (shared_ptr shape)
        const aMeshPtr = meshHandlePtr.readPointer();
        extractAndSendAMesh(aMeshPtr, 'AWM/ingest_type' + type);
      } catch (e) {
        send({ type: 'error', where: 'AWM hook', error: e.toString() });
      }
    },
  });
  console.log('[*] hooked OnlineMergingManager::AWM at ' + addr + ' (RVA 0x' + RVA.toString(16) + ') [EXPERIMENTAL]');
}

// ---------------------------------------------------------------------
// SCAN CONTROL -- fully programmatic start/stop, no hardware button
// ---------------------------------------------------------------------
// CONFIRMED WORKING (2026-08-09, live: input 0->754, publish 0->420). A real
// scan needs BOTH halves:
//   (1) DEVICE streaming commands on the OUTBOUND control socket, AND
//   (2) a synthetic HARDWARE-BUTTON press fed INBOUND into 3ds's OnSocketMsg.
// Device command alone -> green LED but no frames (device never streams).
// Press alone -> 3ds goes green + enables intake but the device never streams.
// Together -> frames flow (IOSAlgo InputTrigger climbs) and meshes publish.
//
// Frame layout (272 bytes): magic "\x00TNI" @0, counter @4, len @12,
// opcode @16, const 2 @24, arg @28, rest zero.
//   Outbound device cmds: 0x105 video-enable, 0x106 metal-scan, 0x104 view-fps.
//   Inbound button press: 0x303 EVT_CAP_BUT_DOWN then 0x304 EVT_CAP_BUT_UP
//     (a full DOWN+UP pair; +12=0x100, +28=2 -- byte-identical to a captured
//     real press). The press is a TOGGLE: start and stop both send one press;
//     only the device commands differ (enable vs disable the stream).
const CTRL_MAGIC = 0x494e5400;    // "\x00TNI" little-endian
const ONSOCKETMSG_RVA = 0x1c54d0; // MixDevice::OnSocketMsg(this, frame, a2, a3)

let _pendingDev = 0;   // 1 start / -1 stop -- device commands, flushed on the OUTBOUND send hook
let _pendingPress = 0; // 1 start / -1 stop -- host press, flushed on the INBOUND OnSocketMsg hook
let _sendNF = null, _onSocketMsg = null;

function _devFrame(counter, opcode, arg) {   // outbound device control frame
  const b = Memory.alloc(272); b.writeByteArray(new Array(272).fill(0));
  b.writeU8(0); b.add(1).writeU8(0x54); b.add(2).writeU8(0x4e); b.add(3).writeU8(0x49); // \x00TNI
  b.add(4).writeU32(counter >>> 0); b.add(12).writeU32(0x10); b.add(16).writeU32(opcode);
  b.add(24).writeU32(2); b.add(28).writeU32(arg >>> 0);
  return b;
}
function _pressFrame(opcode) {                // inbound button-event frame (matches a real press)
  const b = Memory.alloc(272); b.writeByteArray(new Array(272).fill(0));
  b.writeU8(0); b.add(1).writeU8(0x54); b.add(2).writeU8(0x4e); b.add(3).writeU8(0x49);
  b.add(12).writeU32(0x100); b.add(16).writeU32(opcode); b.add(24).writeU32(2); b.add(28).writeU32(2);
  return b;
}

// STEP 1 (outbound send thread): inject the device streaming commands on a real
// heartbeat frame (valid counter + correct socket), then arm the press (STEP 2).
function _maybeInjectDev(sockPtr, buf, len) {
  if (_pendingDev === 0 || len < 32) return;
  let magic; try { magic = buf.readU32(); } catch (e) { return; }
  if (magic !== CTRL_MAGIC) return;
  const c = buf.add(4).readU32();
  const want = _pendingDev; _pendingDev = 0; // clear before our own send()s so they don't re-enter
  try {
    if (want === 1) {
      _sendNF(sockPtr, _devFrame(c, 0x105, 1), 272, 0);             // video enable
      _sendNF(sockPtr, _devFrame((c + 1) >>> 0, 0x106, 0), 272, 0); // metal-scan off
      _sendNF(sockPtr, _devFrame((c + 2) >>> 0, 0x104, 1), 272, 0); // view-fps on
    } else {
      _sendNF(sockPtr, _devFrame(c, 0x105, 0), 272, 0);             // video disable
      _sendNF(sockPtr, _devFrame((c + 1) >>> 0, 0x104, 0), 272, 0); // view-fps off
    }
    send({ type: 'scan_control', ok: true, action: want === 1 ? 'device-start' : 'device-stop' });
    _pendingPress = want; // now fire the host button press (STEP 2)
  } catch (e) { send({ type: 'scan_control', ok: false, error: 'dev: ' + e }); }
}

// STEP 2 (inbound OnSocketMsg thread): inject a full DOWN+UP press using the
// real message's a0/a2/a3 context, so 3ds runs its own complete start/stop and
// resolves its own engine pointer.
function _maybeInjectPress(a0, a2, a3) {
  if (_pendingPress === 0) return;
  _pendingPress = 0;
  try {
    _onSocketMsg(a0, _pressFrame(0x303), a2, a3); // DOWN
    _onSocketMsg(a0, _pressFrame(0x304), a2, a3); // UP
    send({ type: 'scan_control', ok: true, action: 'press' });
  } catch (e) { send({ type: 'scan_control', ok: false, error: 'press: ' + e }); }
}

function installScanControl() {
  const dsMod = Process.findModuleByName('3ds.exe') || Process.mainModule;
  if (!dsMod) { console.log('[scan] 3ds.exe module not found -- scan control disabled'); return; }
  _onSocketMsg = new NativeFunction(dsMod.base.add(ONSOCKETMSG_RVA), 'void', ['pointer', 'pointer', 'pointer', 'pointer']);

  const ws2 = Process.findModuleByName('ws2_32.dll');
  const sendp = ws2 ? ws2.findExportByName('send') : null;
  if (!sendp) { console.log('[scan] ws2_32 send not found -- scan control disabled'); return; }
  _sendNF = new NativeFunction(sendp, 'int', ['pointer', 'pointer', 'int', 'int']);
  Interceptor.attach(sendp, { onEnter(a) { _maybeInjectDev(a[0], a[1], a[2].toInt32()); } });
  const wsasend = ws2 ? ws2.findExportByName('WSASend') : null;
  if (wsasend) {
    Interceptor.attach(wsasend, {
      onEnter(a) {
        if (_pendingDev === 0) return;
        try {
          const cnt = a[2].toInt32(), bufs = a[1];
          for (let i = 0; i < cnt; i++) { const wb = bufs.add(i * 16); _maybeInjectDev(a[0], wb.add(8).readPointer(), wb.readU32()); }
        } catch (e) {}
      },
    });
  }
  // The press rides a real inbound message so a0/a2/a3 context is valid (and the
  // device commands above provoke an inbound reply, so this fires promptly).
  Interceptor.attach(dsMod.base.add(ONSOCKETMSG_RVA), {
    onEnter(a) {
      let op = -1; try { op = a[1].add(16).readU32(); } catch (e) {}
      if (op === 0x303 || op === 0x304) return; // ignore our own injected frames
      _maybeInjectPress(a[0], a[2], a[3]);
    },
  });
  console.log('[scan] scan control ready -- device stream cmds + synthetic button press (fully programmatic)');
}

// RPC: viewer buttons. Start/Stop each fire device commands then a button press.
rpc.exports = {
  startScan: function () { _pendingDev = 1; return true; },
  stopScan: function () { _pendingDev = -1; return true; },
};

function main() {
  const mod = Process.findModuleByName(MODULE_NAME);
  if (!mod) {
    console.log('[!] ' + MODULE_NAME + ' not loaded yet in this process. Attach after the scanner app has finished loading.');
    send({ type: 'status', ok: false, reason: 'module_not_found' });
    return;
  }
  console.log('[*] ' + MODULE_NAME + ' base = ' + mod.base);
  installThreadMPPublishHook(mod.base);
  if (ENABLE_AWM_INGEST_HOOK) installAWMIngestHook(mod.base);
  installScanControl();
  send({ type: 'status', ok: true, module_base: mod.base.toString() });
}

main();
