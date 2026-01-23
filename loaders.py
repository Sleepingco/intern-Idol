# loaders.py - 3D 모델 및 텍스처 로더
import os
import numpy as np
import ctypes
from collections import deque
from OpenGL.GL import *
from pygltflib import GLTF2, TextureInfo
from config import GLB_PATH, MAX_BONES, ROPE_DAMP, ROPE_ITERS


def load_obj_mesh(path):
    """OBJ 파일을 로드하여 메쉬 데이터 반환 (UV 포함)"""
    vs, vts, vns = [], [], []
    faces = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                vs.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("vt "):
                parts = line.strip().split()
                u = float(parts[1]) if len(parts) > 1 else 0.0
                v = float(parts[2]) if len(parts) > 2 else 0.0
                vts.append([u, v])
            elif line.startswith("vn "):
                parts = line.strip().split()
                vns.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                tri = []
                for p in parts:
                    indices = p.split("/")
                    vi = int(indices[0]) - 1 if indices[0] else 0
                    vti = int(indices[1]) - 1 if len(indices) > 1 and indices[1] else 0
                    vni = int(indices[2]) - 1 if len(indices) > 2 and indices[2] else 0
                    tri.append((vi, vti, vni))
                for i in range(1, len(tri) - 1):
                    faces.append([tri[0], tri[i], tri[i + 1]])
    if not vns:
        vns = [[0.0, 0.0, 1.0]] * len(vs)
    if not vts:
        vts = [[0.0, 0.0]] * len(vs)
    verts = []
    for face in faces:
        for vi, vti, vni in face:
            verts.extend(vs[vi])
            verts.extend(vns[vni])
            verts.extend(vts[vti])
    return np.array(verts, dtype=np.float32).reshape(-1, 8)


def _find_mtl_for_obj(obj_path):
    """OBJ 파일에서 MTL 파일 경로 찾기"""
    mtl = None
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("mtllib "):
                mtl = line.strip().split(maxsplit=1)[1]
                break
    if not mtl:
        return None
    if not os.path.isabs(mtl):
        mtl = os.path.join(os.path.dirname(obj_path), mtl)
    return mtl if os.path.exists(mtl) else None


def _extract_tex_path(parts, start_idx=1):
    """MTL 파일에서 텍스처 경로 추출"""
    i = start_idx
    while i < len(parts):
        tok = parts[i]
        if tok.startswith("-"):
            i += 2
            continue
        return _remap_filename(tok)
    return None


def _remap_filename(filename):
    """파일명 매핑"""
    mapping = {
        "DefaultMaterial_Base_color_1001.png": "./obj/DefaultMaterial_BaseColor.1001.png",
        "DefaultMaterial_Roughness_1001.png": "./obj/DefaultMaterial_Roughness.1001.png",
        "DefaultMaterial_Opacity_1001.png": "./obj/DefaultMaterial_Alpha.1001.png",
        "DefaultMaterial_Normal_OpenGL_1001.png": "./obj/DefaultMaterial_Normal.1001.png",
    }
    return mapping.get(filename, filename)


def _load_texture_file(filename):
    """텍스처 파일 로드"""
    filename = _remap_filename(filename)
    if not os.path.exists(filename):
        print(f"텍스처 파일 없음: {filename}")
        return None
    try:
        from PIL import Image

        img = Image.open(filename).convert("RGBA")
        img_data = np.array(img, dtype=np.uint8)
        print(f"텍스처 로딩: {filename} ({img.width}x{img.height})")
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)

        # 개선된 텍스처 설정: 밉맵 활성화 + 고품질 필터링
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        # 밉맵 최대 레벨 제한으로 디테일 유지
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 4)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            img.width,
            img.height,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            img_data,
        )

        # 밉맵 생성으로 거리별 품질 개선
        glGenerateMipmap(GL_TEXTURE_2D)

        glBindTexture(GL_TEXTURE_2D, 0)
        return tex_id
    except Exception as e:
        print(f"텍스처 로드 실패 {filename}: {e}")
        return None


def _parse_mtl_load_textures(mtl_path, obj_dir=None):
    """MTL 파일 파싱하여 텍스처 로드"""
    base = alpha = normal = rough = None
    if obj_dir is None:
        obj_dir = os.path.dirname(mtl_path) if mtl_path else os.getcwd()
    if mtl_path and os.path.exists(mtl_path):
        with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                key = parts[0].lower()
                if key in ("map_kd",):
                    path = _extract_tex_path(parts, 1)
                    if path:
                        base = _load_texture_file(path)
                        print(f"  - Base map: {path}")
                elif key in ("map_d", "map_opacity"):
                    path = _extract_tex_path(parts, 1)
                    if path:
                        alpha = _load_texture_file(path)
                        print(f"  - Alpha map: {path}")
                elif key in (
                    "bump",
                    "map_bump",
                    "map_bump1",
                    "map_bump2",
                    "map_bump3",
                    "map_bump4",
                    "map_bump5",
                    "norm",
                    "normal",
                ):
                    path = _extract_tex_path(parts, 1)
                    if path:
                        normal = _load_texture_file(path)
                        print(f"  - Normal map: {path}")
                elif "rough" in key:
                    path = _extract_tex_path(parts, 1)
                    if path:
                        rough = _load_texture_file(path)
                        print(f"  - Roughness map: {path}")
    # 표준 파일명 자동 보강
    if base is None and os.path.exists("./obj/DefaultMaterial_BaseColor.1001.png"):
        base = _load_texture_file("./obj/DefaultMaterial_BaseColor.1001.png")
    if alpha is None and os.path.exists("./obj/DefaultMaterial_Alpha.1001.png"):
        alpha = _load_texture_file("./obj/DefaultMaterial_Alpha.1001.png")
    if normal is None and os.path.exists("./obj/DefaultMaterial_Normal.1001.png"):
        normal = _load_texture_file("./obj/DefaultMaterial_Normal.1001.png")
    if rough is None and os.path.exists("./obj/DefaultMaterial_Roughness.1001.png"):
        rough = _load_texture_file("./obj/DefaultMaterial_Roughness.1001.png")
    return {"base": base, "alpha": alpha, "normal": normal, "roughness": rough}


def load_textures_for_obj(obj_path):
    """OBJ 파일에 대한 텍스처 로드"""
    mtl = _find_mtl_for_obj(obj_path)
    if mtl:
        print("[MTL]", mtl)
        return _parse_mtl_load_textures(mtl, os.path.dirname(obj_path))
    return _parse_mtl_load_textures(None, os.path.dirname(obj_path))


def bind_mat_textures(tex):
    """재질 텍스처 바인딩"""
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, tex.get("base", 0))
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, tex.get("alpha", 0))
    glActiveTexture(GL_TEXTURE2)
    glBindTexture(GL_TEXTURE_2D, tex.get("normal", 0))
    glActiveTexture(GL_TEXTURE3)
    glBindTexture(GL_TEXTURE_2D, tex.get("roughness", 0))
    glActiveTexture(GL_TEXTURE0)


# -------------------- GLTF 로딩 (스킨 지원) --------------------
def _read_accessor(gltf, accessor_idx):
    """GLTF accessor 데이터 읽기"""
    acc = gltf.accessors[accessor_idx]
    bv = gltf.bufferViews[acc.bufferView]
    buf = gltf.buffers[bv.buffer]
    if buf.uri is None:
        raw = gltf.binary_blob()
    else:
        with open(os.path.join(os.path.dirname(GLB_PATH), buf.uri), "rb") as f:
            raw = f.read()
    start = bv.byteOffset or 0
    end = start + (bv.byteLength or 0)
    view = raw[start:end]
    c2d = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }
    t2n = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT2": 4,
        "MAT3": 9,
        "MAT4": 16,
    }
    dt = c2d[acc.componentType]
    ncomp = t2n[acc.type]
    stride = bv.byteStride or (np.dtype(dt).itemsize * ncomp)
    count = acc.count
    arr = np.empty((count, ncomp), dtype=np.dtype(dt))
    off = acc.byteOffset or 0
    for i in range(count):
        s = off + i * stride
        e = s + np.dtype(dt).itemsize * ncomp
        arr[i] = np.frombuffer(view[s:e], dtype=dt, count=ncomp)
    return arr, acc.componentType


def _node_local_matrix(node):
    """노드의 로컬 변환 행렬 계산"""
    if node.matrix is not None:
        return np.array(node.matrix, np.float32).reshape(4, 4).T
    T = np.eye(4, dtype=np.float32)
    if node.translation is not None:
        T[:3, 3] = np.array(node.translation, np.float32)
    R = np.eye(4, dtype=np.float32)
    if node.rotation is not None:
        qw, qx, qy, qz = (
            node.rotation[3],
            node.rotation[0],
            node.rotation[1],
            node.rotation[2],
        )
        R[:3, :3] = np.array(
            [
                [
                    1 - 2 * (qy * qy + qz * qz),
                    2 * (qx * qy - qz * qw),
                    2 * (qx * qz + qy * qw),
                ],
                [
                    2 * (qx * qy + qz * qw),
                    1 - 2 * (qx * qx + qz * qz),
                    2 * (qy * qz - qx * qw),
                ],
                [
                    2 * (qx * qz - qy * qw),
                    2 * (qy * qz + qx * qw),
                    1 - 2 * (qx * qx + qy * qy),
                ],
            ],
            np.float32,
        )
    S = np.eye(4, dtype=np.float32)
    if node.scale is not None:
        S[0, 0], S[1, 1], S[2, 2] = node.scale
    return (T @ R @ S).astype(np.float32)


def _compute_global_node_mats(gltf):
    """글로벌 노드 변환 행렬 계산"""
    n = len(gltf.nodes or [])
    locals_ = [np.eye(4, dtype=np.float32) for _ in range(n)]
    globals_ = [np.eye(4, dtype=np.float32) for _ in range(n)]
    parent = [-1] * n
    for i, node in enumerate(gltf.nodes or []):
        locals_[i] = _node_local_matrix(node)
        for ch in node.children or []:
            parent[ch] = i
    for i in range(n):
        if parent[i] == -1:

            def dfs(u, G):
                globals_[u] = G @ locals_[u]
                for v in gltf.nodes[u].children or []:
                    dfs(v, globals_[u])

            dfs(i, np.eye(4, dtype=np.float32))
    return locals_, globals_, parent


def _load_textures(gltf):
    """GLTF 텍스처 로드"""
    tex_ids = []
    for tex in gltf.textures or []:
        img = gltf.images[tex.source]
        if img.uri:
            path = os.path.join(os.path.dirname(GLB_PATH), img.uri)
            from PIL import Image

            im = Image.open(path).convert("RGBA")
        else:
            bv = gltf.bufferViews[img.bufferView]
            raw = gltf.binary_blob()
            start = bv.byteOffset or 0
            end = start + (bv.byteLength or 0)
            import io
            from PIL import Image

            im = Image.open(io.BytesIO(raw[start:end])).convert("RGBA")
        tid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tid)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        w, h = im.size
        data = np.array(im, np.uint8)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data
        )
        glGenerateMipmap(GL_TEXTURE_2D)
        tex_ids.append(tid)
    return tex_ids


def load_glb_batches(path):
    """GLB 파일을 배치로 로드"""
    gltf = GLTF2().load(path)
    tex_ids = _load_textures(gltf)
    node_locals, node_globals, parent_all = _compute_global_node_mats(gltf)
    batches = []
    mesh_nodes = []
    for ni, node in enumerate(gltf.nodes or []):
        if node.mesh is not None:
            mesh_nodes.append((ni, node.mesh, node.skin))
    if not mesh_nodes and gltf.meshes:
        mesh_nodes = [(-1, 0, None)]

    def read_accessor(acc_idx):
        return _read_accessor(gltf, acc_idx)

    for node_idx, mi, skin_idx in mesh_nodes:
        mesh = gltf.meshes[mi]
        node_M = (
            node_globals[node_idx] if node_idx >= 0 else np.eye(4, dtype=np.float32)
        )
        skin_meta = None
        if skin_idx is not None:
            skin = gltf.skins[skin_idx]
            if skin.inverseBindMatrices is not None:
                inv, _ = _read_accessor(gltf, skin.inverseBindMatrices)
                inv_bind = [
                    inv[i].reshape(4, 4).T.astype(np.float32)
                    for i in range(inv.shape[0])
                ]
            else:
                inv_bind = [np.eye(4, dtype=np.float32) for _ in (skin.joints or [])]
            joints = skin.joints
            nodeId_to_skinIdx = {node_id: i for i, node_id in enumerate(joints)}
            parent_idx_skin = []
            local_mats_skin = []
            for i, node_id in enumerate(joints):
                pnode = parent_all[node_id]
                parent_idx_skin.append(nodeId_to_skinIdx.get(pnode, -1))
                local_mats_skin.append(node_locals[node_id])
            skin_meta = dict(
                joints=joints,
                inv_bind=np.array(inv_bind, np.float32),
                parent_idx=np.array(parent_idx_skin, np.int32),
                local_mats=np.array(local_mats_skin, np.float32),
            )

        for prim in mesh.primitives:
            attr = prim.attributes
            pos, _ = read_accessor(attr.POSITION)
            pos = pos.astype(np.float32)
            nor = np.zeros_like(pos, np.float32)
            if attr.NORMAL is not None:
                nor, _ = read_accessor(attr.NORMAL)
                nor = nor.astype(np.float32)
            uv = np.zeros((pos.shape[0], 2), np.float32)
            if attr.TEXCOORD_0 is not None:
                uv, _ = read_accessor(attr.TEXCOORD_0)
                uv = uv.astype(np.float32)

            if prim.indices is not None:
                idx, comp = read_accessor(prim.indices)
                idx = idx.reshape(-1)
                if comp == 5121:
                    index_gl_type = GL_UNSIGNED_BYTE
                    idx_np = idx.astype(np.uint8)
                elif comp == 5123:
                    index_gl_type = GL_UNSIGNED_SHORT
                    idx_np = idx.astype(np.uint16)
                else:
                    index_gl_type = GL_UNSIGNED_INT
                    idx_np = idx.astype(np.uint32)
            else:
                idx_np = np.arange(pos.shape[0], dtype=np.uint32)
                index_gl_type = GL_UNSIGNED_INT

            has_skin = (
                skin_meta is not None
                and attr.JOINTS_0 is not None
                and attr.WEIGHTS_0 is not None
            )
            joint_palette = None
            if has_skin:
                jnts, _ = read_accessor(attr.JOINTS_0)
                wgts, _ = read_accessor(attr.WEIGHTS_0)
                used = np.unique(jnts.reshape(-1)).astype(np.int32)
                if used.size > MAX_BONES:
                    counts = np.bincount(jnts.reshape(-1).astype(np.int32))
                    top = np.argsort(counts)[::-1]
                    joint_palette = [int(i) for i in top if counts[i] > 0][:MAX_BONES]
                else:
                    joint_palette = [int(i) for i in used]
                remap = np.full(max(int(used.max()) + 1, MAX_BONES), -1, np.int32)
                for new_i, old_i in enumerate(joint_palette):
                    remap[old_i] = new_i
                jf = jnts.reshape(-1, 4).astype(np.int32)
                wf = wgts.reshape(-1, 4).astype(np.float32)
                mask = remap[jf] < 0
                wf[mask] = 0.0
                jf = np.where(mask, 0, remap[jf])
                s = np.sum(wf, axis=1, keepdims=True)
                wf = np.where(
                    s <= 1e-8, np.array([[1, 0, 0, 0]], np.float32), wf / (s + 1e-8)
                )
                j_u16 = jf.astype(np.uint16, copy=False)
                w_f32 = wf.astype(np.float32, copy=False)

                stride = 56
                dt = np.dtype(
                    [
                        ("p", np.float32, 3),
                        ("n", np.float32, 3),
                        ("uv", np.float32, 2),
                        ("j", np.uint16, 4),
                        ("w", np.float32, 4),
                    ]
                )
                data = np.zeros(pos.shape[0], dtype=dt)
                data["p"] = pos
                data["n"] = nor
                data["uv"] = uv
                data["j"] = j_u16
                data["w"] = w_f32
            else:
                stride = 32
                dt = np.dtype(
                    [("p", np.float32, 3), ("n", np.float32, 3), ("uv", np.float32, 2)]
                )
                data = np.zeros(pos.shape[0], dtype=dt)
                data["p"] = pos
                data["n"] = nor
                data["uv"] = uv

            vao = glGenVertexArrays(1)
            glBindVertexArray(vao)
            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
            ebo = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx_np.nbytes, idx_np, GL_STATIC_DRAW)

            glEnableVertexAttribArray(0)
            glVertexAttribPointer(
                0,
                3,
                GL_FLOAT,
                GL_FALSE,
                stride,
                ctypes.c_void_p(data.dtype.fields["p"][1]),
            )
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(
                1,
                3,
                GL_FLOAT,
                GL_FALSE,
                stride,
                ctypes.c_void_p(data.dtype.fields["n"][1]),
            )
            glEnableVertexAttribArray(2)
            glVertexAttribPointer(
                2,
                2,
                GL_FLOAT,
                GL_FALSE,
                stride,
                ctypes.c_void_p(data.dtype.fields["uv"][1]),
            )
            if has_skin:
                glEnableVertexAttribArray(3)
                glVertexAttribIPointer(
                    3,
                    4,
                    GL_UNSIGNED_SHORT,
                    stride,
                    ctypes.c_void_p(data.dtype.fields["j"][1]),
                )
                glEnableVertexAttribArray(4)
                glVertexAttribPointer(
                    4,
                    4,
                    GL_FLOAT,
                    GL_FALSE,
                    stride,
                    ctypes.c_void_p(data.dtype.fields["w"][1]),
                )

            baseColorFactor = (1, 1, 1, 1)
            metallicFactor = 0.0
            roughnessFactor = 1.0
            tex_id = None
            metallic_roughness_tex_id = None
            if prim.material is not None:
                m = gltf.materials[prim.material]
                pmr = m.pbrMetallicRoughness
                if pmr:
                    if pmr.baseColorFactor:
                        baseColorFactor = tuple(pmr.baseColorFactor)
                    if pmr.metallicFactor is not None:
                        metallicFactor = pmr.metallicFactor
                    if pmr.roughnessFactor is not None:
                        roughnessFactor = pmr.roughnessFactor
                    if isinstance(pmr.baseColorTexture, TextureInfo):
                        tidx = pmr.baseColorTexture.index
                        if tidx is not None and tidx < len(tex_ids):
                            tex_id = tex_ids[tidx]
                    if isinstance(pmr.metallicRoughnessTexture, TextureInfo):
                        tidx = pmr.metallicRoughnessTexture.index
                        if tidx is not None and tidx < len(tex_ids):
                            metallic_roughness_tex_id = tex_ids[tidx]

            batches.append(
                dict(
                    vao=vao,
                    count=idx_np.shape[0],
                    index_type=index_gl_type,
                    color=baseColorFactor,
                    tex=tex_id,
                    metallicFactor=metallicFactor,
                    roughnessFactor=roughnessFactor,
                    metallic_roughness_tex=metallic_roughness_tex_id,
                    has_skin=1 if has_skin else 0,
                    node_M=node_M,
                    skin_meta=skin_meta,
                    joint_palette=joint_palette,
                )
            )
            glBindVertexArray(0)

    # 스켈레톤 캐시 + 로프 초기화
    for b in batches:
        sm = b.get("skin_meta")
        if sm is None:
            continue
        parents = sm["parent_idx"]
        Ns = int(parents.shape[0])
        children = [[] for _ in range(Ns)]
        for i, p in enumerate(parents):
            if p >= 0:
                children[p].append(i)
        roots = [i for i, p in enumerate(parents) if p < 0]
        order = []
        q = deque(roots)
        while q:
            u = q.popleft()
            order.append(u)
            for v in children[u]:
                q.append(v)
        globals_bind = [np.eye(4, dtype=np.float32) for _ in range(Ns)]
        for i in order:
            p = parents[i]
            if p < 0:
                globals_bind[i] = sm["local_mats"][i]
            else:
                globals_bind[i] = globals_bind[p] @ sm["local_mats"][i]
        G_bind_all = np.stack(globals_bind, axis=0).astype(np.float32)
        P_bind_all = np.stack([g[:3, 3] for g in globals_bind], axis=0).astype(
            np.float32
        )
        b["globals_bind_all_model"] = G_bind_all
        b["P_bind_all_model"] = P_bind_all

        root = roots[0] if roots else 0
        seq = [root]
        cur = root
        while len(children[cur]) == 1:
            cur = children[cur][0]
            seq.append(cur)

        weighted = np.zeros(Ns, dtype=bool)
        pal = b.get("joint_palette") or []
        if len(pal) > 0:
            weighted[np.array(pal, np.int32)] = True
        s = 0
        e = len(seq) - 1
        while s <= e and not weighted[seq[s]]:
            s += 1
        while e >= s and not weighted[seq[e]]:
            e -= 1
        seq_sim = seq[s : e + 1] if e >= s else seq
        b["seq"] = np.array(seq_sim, np.int32)

        P0 = P_bind_all[seq_sim]
        Lseg = np.linalg.norm(P0[1:] - P0[:-1], axis=1).astype(np.float32)
        last_vec = P0[-1] - P0[-2]
        tip_pos = P0[-1] + last_vec
        P_ext_model = np.vstack([P0, tip_pos])  # (N+1,3)
        L_ext = np.concatenate([Lseg, [np.linalg.norm(last_vec)]])
        b["P_world"] = None
        b["P_prev_world"] = None
        b["P_ext_model"] = P_ext_model
        b["Lseg"] = L_ext
        b["Lseg_orig"] = L_ext.copy()
        b["N_joints"] = P0.shape[0]
        b["rope_damp"] = ROPE_DAMP
        b["rope_iters"] = ROPE_ITERS
    return batches
