"""メッシュエクスポートモジュール。

シャドーボックスメッシュをSTL、OBJ、PLY形式でエクスポートする
機能を提供します。
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np

from shadowbox.core.mesh import ShadowboxMesh


def export_to_stl(
    mesh: ShadowboxMesh,
    filepath: Union[str, Path],
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

    # 全レイヤーの頂点と色を収集
    all_vertices = []
    all_colors = []

    for layer in mesh.layers:
        if len(layer.vertices) > 0:
            all_vertices.append(layer.vertices)
            all_colors.append(layer.colors)

    if not all_vertices:
        raise ValueError("メッシュに頂点がありません")

    vertices = np.vstack(all_vertices)

    # 各ポイントを小さな四角形に変換
    quads = _points_to_quads(vertices, size=0.01)

    # フレームがある場合は追加
    if mesh.frame is not None:
        frame_triangles = _frame_to_triangles(mesh.frame)
        quads = np.vstack([quads, frame_triangles])

    if binary:
        _write_binary_stl(filepath, quads)
    else:
        _write_ascii_stl(filepath, quads)

    print(f"STLエクスポート完了: {filepath}")
    print(f"  三角形数: {len(quads)}")


def export_to_obj(
    mesh: ShadowboxMesh,
    filepath: Union[str, Path],
    include_colors: bool = True,
) -> None:
    """メッシュをOBJ形式でエクスポート。

    頂点カラーを含むOBJファイルを出力します。
    MTLファイルは生成せず、頂点カラーをOBJ拡張形式で埋め込みます。

    Args:
        mesh: エクスポートするShadowboxMesh。
        filepath: 出力ファイルパス。
        include_colors: 頂点カラーを含めるかどうか。

    Example:
        >>> export_to_obj(result.mesh, "shadowbox.obj")
    """
    filepath = Path(filepath)

    lines = ["# Shadowbox Generator OBJ Export"]
    lines.append(f"# Layers: {mesh.num_layers}")
    lines.append(f"# Total vertices: {mesh.total_vertices}")
    lines.append("")

    vertex_offset = 1  # OBJは1始まり

    # 各レイヤーを出力
    for layer_idx, layer in enumerate(mesh.layers):
        if len(layer.vertices) == 0:
            continue

        lines.append(f"# Layer {layer_idx}")
        lines.append(f"o Layer{layer_idx}")

        # 頂点を出力（色付き）
        for i, (v, c) in enumerate(zip(layer.vertices, layer.colors)):
            if include_colors:
                # OBJ拡張形式: v x y z r g b（色は0-1に正規化）
                r, g, b = c[0] / 255.0, c[1] / 255.0, c[2] / 255.0
                lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {r:.4f} {g:.4f} {b:.4f}")
            else:
                lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

        # ポイントを小さな四角形に変換して面を出力
        num_points = len(layer.vertices)
        for i in range(num_points):
            # 各ポイントを独立した点として出力
            lines.append(f"p {vertex_offset + i}")

        vertex_offset += num_points
        lines.append("")

    # フレームを出力
    if mesh.frame is not None:
        lines.append("# Frame")
        lines.append("o Frame")

        frame = mesh.frame
        for v in frame.vertices:
            r, g, b = frame.color[0] / 255.0, frame.color[1] / 255.0, frame.color[2] / 255.0
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {r:.4f} {g:.4f} {b:.4f}")

        for face in frame.faces:
            # OBJは1始まりインデックス
            f0, f1, f2 = face[0] + vertex_offset, face[1] + vertex_offset, face[2] + vertex_offset
            lines.append(f"f {f0} {f1} {f2}")

        lines.append("")

    # ファイルに書き込み
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"OBJエクスポート完了: {filepath}")
    print(f"  頂点数: {mesh.total_vertices}")


def export_to_ply(
    mesh: ShadowboxMesh,
    filepath: Union[str, Path],
    binary: bool = False,
) -> None:
    """メッシュをPLY形式でエクスポート。

    頂点カラー付きのポイントクラウドとしてエクスポートします。
    MeshLabやCloudCompareで表示可能です。

    Args:
        mesh: エクスポートするShadowboxMesh。
        filepath: 出力ファイルパス。
        binary: バイナリ形式で出力するかどうか。

    Example:
        >>> export_to_ply(result.mesh, "shadowbox.ply")
    """
    filepath = Path(filepath)

    # 全頂点と色を収集
    all_vertices = []
    all_colors = []

    for layer in mesh.layers:
        if len(layer.vertices) > 0:
            all_vertices.append(layer.vertices)
            all_colors.append(layer.colors)

    if not all_vertices:
        raise ValueError("メッシュに頂点がありません")

    vertices = np.vstack(all_vertices)
    colors = np.vstack(all_colors)

    # フレーム頂点を追加
    frame_vertex_count = 0
    if mesh.frame is not None:
        frame = mesh.frame
        frame_colors = np.tile(frame.color, (len(frame.vertices), 1))
        vertices = np.vstack([vertices, frame.vertices])
        colors = np.vstack([colors, frame_colors])
        frame_vertex_count = len(frame.vertices)

    num_vertices = len(vertices)

    # PLYヘッダー
    header = [
        "ply",
        "format ascii 1.0" if not binary else "format binary_little_endian 1.0",
        f"element vertex {num_vertices}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]

    # フレームの面を追加
    if mesh.frame is not None:
        num_faces = len(mesh.frame.faces)
        header.append(f"element face {num_faces}")
        header.append("property list uchar int vertex_indices")

    header.append("end_header")

    if binary:
        _write_binary_ply(filepath, header, vertices, colors, mesh.frame)
    else:
        _write_ascii_ply(filepath, header, vertices, colors, mesh.frame, num_vertices - frame_vertex_count)

    print(f"PLYエクスポート完了: {filepath}")
    print(f"  頂点数: {num_vertices}")


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

    for v, c in zip(vertices, colors):
        lines.append(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}")

    # フレームの面
    if frame is not None:
        for face in frame.faces:
            # フレーム頂点のオフセットを加算
            f0, f1, f2 = face[0] + point_vertex_count, face[1] + point_vertex_count, face[2] + point_vertex_count
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
        for v, c in zip(vertices, colors):
            f.write(struct.pack("<3f3B", v[0], v[1], v[2], int(c[0]), int(c[1]), int(c[2])))

        # フレームの面
        if frame is not None:
            point_count = len(vertices) - len(frame.vertices)
            for face in frame.faces:
                f0, f1, f2 = face[0] + point_count, face[1] + point_count, face[2] + point_count
                f.write(struct.pack("<B3i", 3, f0, f1, f2))
