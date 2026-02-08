# TCG Shadowbox Generator

[English version](README.md)

TCGカードのイラストをAI深度推定で解析し、インタラクティブな3Dシャドーボックスに変換します。

## 特徴

- **深度推定**: Depth Anything v2 を使用（軽量・高精度、Apache 2.0 ライセンス）
- **自動/手動領域選択**: 10種のアンサンブル検出メソッドによる信頼度付き自動検出、または手動選択
- **レイヤークラスタリング**: シルエット分析による最適レイヤー数の自動探索
- **等高線カットモード**: 生深度の等高線でレイヤー形状を切り取り（手前は狭く、奥は広い自然な形状）
- **メッシュエクスポート**: STL、OBJ、PLY形式での出力（3Dプリントや外部ツール連携）
- **GUIアプリ（PyQt6）**: 領域選択、パラメータ調整、3Dプレビュー、メッシュエクスポート対応。日英バイリンガルUI
- **インタラクティブ3D表示**: マウスで回転・ズームして3Dシャドーボックスを探索
- **複数の入力方法**: URLまたはローカルディレクトリのギャラリーから読み込み
- **テンプレートシステム**: カードテンプレートの保存・再利用（ポケモン、MTG など）

## インストール

```bash
# リポジトリをクローン
git clone https://github.com/2ROSSO/shadowbox-generator.git
cd shadowbox-generator

# uv でインストール（推奨）
uv sync

# または pip でインストール
pip install -e .
```

### オプション依存関係

```bash
# Jupyter Notebook サポート
uv sync --extra jupyter

# スタンドアロン GUI アプリ（PyQt6）
uv sync --extra gui

# すべてのオプション
uv sync --all-extras
```

| オプション | 含まれるパッケージ                 | 用途                                                      |
| ---------- | ---------------------------------- | --------------------------------------------------------- |
| `jupyter`  | jupyter, ipykernel, ipympl, plotly | Jupyter Notebook での実行、**手動領域選択**、**3Dビュー** |
| `gui`      | PyQt6                              | スタンドアロンGUIアプリ                                   |
| `all`      | jupyter + gui                      | すべての安定機能                                          |

> **Note**: Jupyter Notebook で以下の機能を使用するには `jupyter` オプションが必要です:
>
> - **手動領域選択**: `ipympl` により `%matplotlib widget` バックエンドでインタラクティブな操作が可能
> - **3Dビュー**: `plotly` により `render_shadowbox()` がセル内でインタラクティブ3D表示されます

## クイックスタート

### Jupyter Notebook で使用

```bash
uv run jupyter lab
# notebooks/01_quickstart.ipynb を開く
```

### Python で使用

```python
from PIL import Image
from shadowbox import create_pipeline, ShadowboxSettings

# 画像を読み込み
image = Image.open("card.png")

# デフォルト設定でパイプラインを作成
pipeline = create_pipeline()

# 画像を処理（イラスト領域を自動検出）
result = pipeline.process(image, auto_detect=True)

# 3Dシャドーボックスを表示
from shadowbox.visualization import ShadowboxRenderer
renderer = ShadowboxRenderer(ShadowboxSettings().render)
renderer.render(result.mesh)
```

### GUI アプリで使用

```bash
uv run python -m shadowbox.gui.app
```

## 仕組み

1. **画像読み込み**: URLまたはローカルディレクトリから読み込み
2. **領域選択**: イラスト領域の自動検出または手動指定
3. **深度推定**: AIモデルがピクセルごとの深度を推定
4. **クラスタリング**: 深度値を離散的なレイヤーに分割
5. **マスクモード**: クラスタラベル（平板レイヤー）または等高線カット（深度ベースの自然な形状）を選択
6. **3D生成**: レイヤーを積み重ねた3Dサーフェスとしてレンダリング
7. **インタラクティブ表示**: マウスの回転・ズームで探索

## アーキテクチャ

TCG Shadowbox Generator は2つの3D生成モードをサポートしています：

### Depth モード（デフォルト）
画像から深度マップを推定し、クラスタリングでレイヤーに分割します。

```
Image → Depth Estimation → DepthToMeshProcessor → ShadowboxMesh → 3D View
              ↓                    ↓
          depth map      クラスタリング + メッシュ生成
```

### TripoSR モード
単一画像から直接3Dメッシュを生成し、深度復元でレイヤー化します。

```
Image → TripoSR → 3D Mesh → Depth Recovery → DepthToMeshProcessor → ShadowboxMesh
                     ↓            ↓                    ↓
                 trimesh     depth map       クラスタリング + メッシュ生成
```

**共通コンポーネント**: 両モードは `DepthToMeshProcessor` に収束し、クラスタリング（`LayerClusterer`）とメッシュ生成を共有しています。Frame/BackPanel ファクトリーも共通です。

詳細なアーキテクチャ分析は `docs/architecture_analysis.md` を参照してください。

## プロジェクト構成

```text
shadowbox-generator/
├── src/shadowbox/
│   ├── core/           # パイプライン、深度、クラスタリング、メッシュ
│   ├── config/         # 設定とテンプレート
│   ├── triposr/        # TripoSR統合（生成器、深度復元、メッシュ分割）
│   ├── visualization/  # 2D/3Dレンダリング
│   ├── detection/      # 自動領域検出（10種のアンサンブルメソッド）
│   ├── gui/            # PyQt6 GUIアプリ（ウィジェットタブ構成）
│   └── utils/          # 画像読み込み・処理ユーティリティ
├── notebooks/          # Jupyter ノートブック
├── data/templates/     # カードテンプレート（YAML）
├── docs/               # アーキテクチャドキュメント
└── tests/              # テストスイート
```

## 深度モデルの切り替え

深度推定器はプラグイン方式で差し替え可能です。現在サポートされているモデル：

- **Depth Anything v2**（デフォルト）: Apache 2.0、軽量・高精度
- **MiDaS**: MIT ライセンス、成熟したエコシステム
- **ZoeDepth**: MIT ライセンス、メトリック深度対応

```python
from shadowbox.config import ShadowboxSettings

settings = ShadowboxSettings()
settings.depth.model_type = "midas"  # モデルを切り替え
pipeline = create_pipeline(settings)
```

## 開発

```bash
# テスト実行
uv run pytest

# リンター実行
uv run ruff check src/

# 型チェック実行
uv run mypy src/
```

## TripoSR 統合（試験搭載）

> **注意**: この機能は試験搭載です。TripoSR の依存関係（`uv sync --extra triposr`）をインストールすると、既存の依存関係と競合する場合があります（例: `opencv-python` vs `opencv-python-headless`）。別の仮想環境での利用を推奨します。

TripoSR を使用して単一画像から直接3Dメッシュを生成するには、以下の手順でセットアップが必要です：

### 前提条件（Windows）

- **Visual Studio Build Tools** が必要です（C++拡張のビルドに使用）
- [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) から「C++ によるデスクトップ開発」をインストール

### インストール手順

```bash
# 1. 依存関係をインストール
uv sync --extra triposr

# 2. TripoSR をクローン（プロジェクトルートに配置）
git clone https://github.com/VAST-AI-Research/TripoSR.git

# 3. torchmcubes をビルド・インストール（C++拡張）
uv pip install scikit-build-core cmake ninja pybind11

# Windows の場合、CMAKE_PREFIX_PATH を設定してビルド
set CMAKE_PREFIX_PATH=.venv\Lib\site-packages\torch\share\cmake
uv pip install git+https://github.com/tatsy/torchmcubes.git --no-build-isolation
```

> **Note**: `TripoSR/` ディレクトリはプロジェクトルートに配置すると自動検出されます。
> PYTHONPATH の手動設定は不要です。

### 使用方法

```python
from shadowbox import create_pipeline, ShadowboxSettings

settings = ShadowboxSettings()
settings.model_mode = "triposr"
pipeline = create_pipeline(settings)
result = pipeline.process(image)
```

### 注意事項

- `rembg`（背景除去）は `triposr` オプションに含まれています
- 初回実行時にモデル（約1GB）がダウンロードされます
- GPU（CUDA）推奨ですが、CPUでも動作します（処理速度は遅くなります）

## ライセンス

MIT License
