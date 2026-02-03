"""メッシュエクスポートモジュール。

シャドーボックスメッシュをSTL、OBJ、PLY形式でエクスポートする
機能を提供します。
"""

from pathlib import Path

import numpy as np

from shadowbox.core.mesh import ShadowboxMesh


def export_to_stl(
    mesh: ShadowboxMesh,
    filepath: str | Path,
    binary: bool = True,
) -> None:
    """メッシュをSTL形式でエクスポート。

    ポイントクラウドを小さな四角形（quad）として出力し、
    3Dプリンタやビューアで表示可能にします。

    Args:
        mesh: エクスポートするShadowboxMesh。
        filepath: 出力ファイルパス。
        binary: バイナリSTL形式で出力するかどうか。

    Example:
        >>> export_to_stl(result.mesh, "shadowbox.stl")
    """
    filepath = Path(filepath)

    all_triangles = []

    for layer in mesh.layers:
        if len(layer.vertices) == 0:
            continue

        if layer.faces is not None:
            # 三角形メッシュを直接使用（TripoSR等）
            layer_triangles = _layer_faces_to_triangles(layer)
            all_triangles.append(layer_triangles)
        else:
            # ポイントを四角形に変換（従来のシャドーボックス）
            quads = _points_to_quads(layer.vertices, size=0.01)
            all_triangles.append(quads)

    if not all_triangles:
        raise ValueError("メッシュに頂点がありません")

    triangles = np.vstack(all_triangles)

    # フレームがある場合は追加
    if mesh.frame is not None:
        frame_triangles = _frame_to_triangles(mesh.frame)
        triangles = np.vstack([triangles, frame_triangles])

    if binary:
        _write_binary_stl(filepath, triangles)
    else:
        _write_ascii_stl(filepath, triangles)

    print(f"STLエクスポート完了: {filepath}")
    print(f"  三角形数: {len(triangles)}")


def export_to_obj(
    mesh: ShadowboxMesh,
    filepath: str | Path,
    include_colors: bool = True,
    point_size: float = 0.008,
) -> None:
    """メッシュをOBJ形式でエクスポート。

    各ポイントを小さな四角形（2三角形）としてエクスポートし、
    Blender等の3Dソフトで表示可能にします。

    Args:
        mesh: エクスポートするShadowboxMesh。
        filepath: 出力ファイルパス。
        include_colors: 頂点カラーを含めるかどうか。
        point_size: 各ポイントを表す四角形のサイズ。

    Example:
        >>> export_to_obj(result.mesh, "shadowbox.obj")
    """
    filepath = Path(filepath)

    lines = ["# Shadowbox Generator OBJ Export"]
    lines.append(f"# Layers: {mesh.num_layers}")
    lines.append(f"# Total vertices: {mesh.total_vertices}")
    lines.append("# Each point is rendered as a small quad (2 triangles)")
    lines.append("")

    vertex_offset = 1  # OBJは1始まり
    half = point_size / 2

    # 各レイヤーを出力
    total_faces = 0
    for layer_idx, layer in enumerate(mesh.layers):
        if len(layer.vertices) == 0:
            continue

        lines.append(f"# Layer {layer_idx}")
        lines.append(f"o Layer{layer_idx}")

        if layer.faces is not None:
            # 三角形メッシュを直接出力（TripoSR等）
            for v, c in zip(layer.vertices, layer.colors, strict=True):
                if include_colors:
                    r, g, b = c[0] / 255.0, c[1] / 255.0, c[2] / 255.0
                    color_str = f" {r:.4f} {g:.4f} {b:.4f}"
                else:
                    color_str = ""
                lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}{color_str}")

            for face in layer.faces:
                f0 = face[0] + vertex_offset
                f1 = face[1] + vertex_offset
                f2 = face[2] + vertex_offset
                lines.append(f"f {f0} {f1} {f2}")

            vertex_offset += len(layer.vertices)
            total_faces += len(layer.faces)
        else:
            # 各ポイントを4頂点の四角形として出力
            for v, c in zip(layer.vertices, layer.colors, strict=True):
                if include_colors:
                    r, g, b = c[0] / 255.0, c[1] / 255.0, c[2] / 255.0
                    color_str = f" {r:.4f} {g:.4f} {b:.4f}"
                else:
                    color_str = ""

                # 四角形の4頂点
                lines.append(f"v {v[0] - half:.6f} {v[1] - half:.6f} {v[2]:.6f}{color_str}")
                lines.append(f"v {v[0] + half:.6f} {v[1] - half:.6f} {v[2]:.6f}{color_str}")
                lines.append(f"v {v[0] + half:.6f} {v[1] + half:.6f} {v[2]:.6f}{color_str}")
                lines.append(f"v {v[0] - half:.6f} {v[1] + half:.6f} {v[2]:.6f}{color_str}")

            # 面を出力（2三角形で四角形を構成）
            num_points = len(layer.vertices)
            for i in range(num_points):
                base = vertex_offset + i * 4
                # 三角形1: 0-1-2
                lines.append(f"f {base} {base + 1} {base + 2}")
                # 三角形2: 0-2-3
                lines.append(f"f {base} {base + 2} {base + 3}")

            vertex_offset += num_points * 4
            total_faces += num_points * 2

        lines.append("")

    # フレームを出力
    if mesh.frame is not None:
        lines.append("# Frame")
        lines.append("o Frame")

        frame = mesh.frame
        r, g, b = frame.color[0] / 255.0, frame.color[1] / 255.0, frame.color[2] / 255.0
        color_str = f" {r:.4f} {g:.4f} {b:.4f}" if include_colors else ""

        for v in frame.vertices:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}{color_str}")

        for face in frame.faces:
            f0, f1, f2 = face[0] + vertex_offset, face[1] + vertex_offset, face[2] + vertex_offset
            lines.append(f"f {f0} {f1} {f2}")

        lines.append("")

    # ファイルに書き込み
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    total_faces += len(mesh.frame.faces) if mesh.frame else 0

    print(f"OBJエクスポート完了: {filepath}")
    print(f"  頂点数: {mesh.total_vertices}")
    print(f"  面数: {total_faces}")


def export_to_ply(
    mesh: ShadowboxMesh,
    filepath: str | Path,
    point_size: float = 0.008,
) -> None:
    """メッシュをPLY形式でエクスポート。

    各ポイントを小さな四角形（2三角形）としてエクスポートし、
    Blender等の3Dソフトで表示可能にします。

    Args:
        mesh: エクスポートするShadowboxMesh。
        filepath: 出力ファイルパス。
        point_size: 各ポイントを表す四角形のサイズ。

    Example:
        >>> export_to_ply(result.mesh, "shadowbox.ply")
    """
    filepath = Path(filepath)
    half = point_size / 2

    # 各ポイントを4頂点の四角形に展開
    all_vertices = []
    all_colors = []
    all_faces = []
    vertex_offset = 0

    for layer in mesh.layers:
        if len(layer.vertices) == 0:
            continue

        if layer.faces is not None:
            # 三角形メッシュを直接使用（TripoSR等）
            for v in layer.vertices:
                all_vertices.append([v[0], v[1], v[2]])
            for c in layer.colors:
                all_colors.append(c)
            for face in layer.faces:
                all_faces.append([
                    face[0] + vertex_offset,
                    face[1] + vertex_offset,
                    face[2] + vertex_offset
                ])
            vertex_offset += len(layer.vertices)
        else:
            # ポイントを四角形に展開（従来のシャドーボックス）
            for v, c in zip(layer.vertices, layer.colors, strict=True):
                # 四角形の4頂点
                all_vertices.append([v[0] - half, v[1] - half, v[2]])
                all_vertices.append([v[0] + half, v[1] - half, v[2]])
                all_vertices.append([v[0] + half, v[1] + half, v[2]])
                all_vertices.append([v[0] - half, v[1] + half, v[2]])

                # 4頂点に同じ色
                for _ in range(4):
                    all_colors.append(c)

                # 2つの三角形面
                base = vertex_offset
                all_faces.append([base, base + 1, base + 2])
                all_faces.append([base, base + 2, base + 3])
                vertex_offset += 4

    if not all_vertices:
        raise ValueError("メッシュに頂点がありません")

    # フレーム頂点と面を追加
    if mesh.frame is not None:
        frame = mesh.frame
        frame_base = vertex_offset

        for v in frame.vertices:
            all_vertices.append([v[0], v[1], v[2]])
            all_colors.append(frame.color)

        for face in frame.faces:
            all_faces.append([
                face[0] + frame_base,
                face[1] + frame_base,
                face[2] + frame_base
            ])

    num_vertices = len(all_vertices)
    num_faces = len(all_faces)

    # PLYファイルを書き込み
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {num_vertices}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        f"element face {num_faces}",
        "property list uchar int vertex_indices",
        "end_header",
    ]

    # 頂点データ
    for v, c in zip(all_vertices, all_colors, strict=True):
        lines.append(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}")

    # 面データ
    for face in all_faces:
        lines.append(f"3 {face[0]} {face[1]} {face[2]}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"PLYエクスポート完了: {filepath}")
    print(f"  頂点数: {num_vertices}")
    print(f"  面数: {num_faces}")


def _points_to_quads(vertices: np.ndarray, size: float = 0.01) -> np.ndarray:
    """ポイントを小さな四角形（2三角形）に変換。"""
    n = len(vertices)
    triangles = np.zeros((n * 2, 3, 3), dtype=np.float32)

    half = size / 2

    for i, v in enumerate(vertices):
        # 四角形の4頂点
        p0 = [v[0] - half, v[1] - half, v[2]]
        p1 = [v[0] + half, v[1] - half, v[2]]
        p2 = [v[0] + half, v[1] + half, v[2]]
        p3 = [v[0] - half, v[1] + half, v[2]]

        # 2つの三角形
        triangles[i * 2] = [p0, p1, p2]
        triangles[i * 2 + 1] = [p0, p2, p3]

    return triangles


def _layer_faces_to_triangles(layer) -> np.ndarray:
    """レイヤーメッシュの面を三角形配列に変換（TripoSR用）。"""
    triangles = np.zeros((len(layer.faces), 3, 3), dtype=np.float32)

    for i, face in enumerate(layer.faces):
        triangles[i] = [
            layer.vertices[face[0]],
            layer.vertices[face[1]],
            layer.vertices[face[2]],
        ]

    return triangles


def _frame_to_triangles(frame) -> np.ndarray:
    """フレームメッシュを三角形配列に変換。"""
    triangles = np.zeros((len(frame.faces), 3, 3), dtype=np.float32)

    for i, face in enumerate(frame.faces):
        triangles[i] = [
            frame.vertices[face[0]],
            frame.vertices[face[1]],
            frame.vertices[face[2]],
        ]

    return triangles


def _write_binary_stl(filepath: Path, triangles: np.ndarray) -> None:
    """バイナリSTLを書き込み。"""
    import struct

    with open(filepath, "wb") as f:
        # 80バイトヘッダー
        f.write(b"\0" * 80)

        # 三角形数
        f.write(struct.pack("<I", len(triangles)))

        for tri in triangles:
            # 法線（簡易計算）
            v0, v1, v2 = tri
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            else:
                normal = np.array([0, 0, 1], dtype=np.float32)

            # 法線
            f.write(struct.pack("<3f", *normal))

            # 3頂点
            for v in tri:
                f.write(struct.pack("<3f", *v))

            # 属性バイト
            f.write(struct.pack("<H", 0))


def _write_ascii_stl(filepath: Path, triangles: np.ndarray) -> None:
    """ASCII STLを書き込み。"""
    lines = ["solid shadowbox"]

    for tri in triangles:
        v0, v1, v2 = tri
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        else:
            normal = np.array([0, 0, 1])

        lines.append(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}")
        lines.append("    outer loop")
        for v in tri:
            lines.append(f"      vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        lines.append("    endloop")
        lines.append("  endfacet")

    lines.append("endsolid shadowbox")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_ascii_ply(
    filepath: Path,
    header: list,
    vertices: np.ndarray,
    colors: np.ndarray,
    frame,
    point_vertex_count: int,
) -> None:
    """ASCII PLYを書き込み。"""
    lines = header.copy()

    for v, c in zip(vertices, colors, strict=True):
        lines.append(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}")

    # フレームの面
    if frame is not None:
        for face in frame.faces:
            # フレーム頂点のオフセットを加算
            offset = point_vertex_count
            f0, f1, f2 = face[0] + offset, face[1] + offset, face[2] + offset
            lines.append(f"3 {f0} {f1} {f2}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_binary_ply(
    filepath: Path,
    header: list,
    vertices: np.ndarray,
    colors: np.ndarray,
    frame,
) -> None:
    """バイナリPLYを書き込み。"""
    import struct

    with open(filepath, "wb") as f:
        # ヘッダー（ASCII）
        f.write(("\n".join(header) + "\n").encode("utf-8"))

        # 頂点データ
        for v, c in zip(vertices, colors, strict=True):
            f.write(struct.pack("<3f3B", v[0], v[1], v[2], int(c[0]), int(c[1]), int(c[2])))

        # フレームの面
        if frame is not None:
            point_count = len(vertices) - len(frame.vertices)
            for face in frame.faces:
                f0, f1, f2 = face[0] + point_count, face[1] + point_count, face[2] + point_count
                f.write(struct.pack("<B3i", 3, f0, f1, f2))
