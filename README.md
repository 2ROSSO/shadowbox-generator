# TCG Shadowbox Generator

Transform TCG card illustrations into interactive 3D shadowbox displays using AI-powered depth estimation.

## Features

- **Depth Estimation**: Uses Depth Anything v2 (lightweight, Apache 2.0 license)
- **Auto/Manual Region Selection**: Automatically detect or manually select illustration areas
- **Layer Clustering**: Automatically finds optimal number of layers using silhouette analysis
- **Contour Cut Mode**: Depth-based contour masking for natural layer shapes (front layers narrow, back layers wide)
- **Interactive 3D View**: Rotate and explore shadowbox with mouse
- **Multiple Input Methods**: Load from URL or select from local directory gallery
- **Template System**: Save and reuse card templates (Pokemon, MTG, etc.)

## Installation

```bash
# Clone the repository
git clone https://github.com/2ROSSO/shadowbox-generator.git
cd shadowbox-generator

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

### Optional Dependencies

```bash
# For Jupyter notebook support
uv sync --extra jupyter

# For standalone GUI app (PyQt6)
uv sync --extra gui

# For everything
uv sync --all-extras
```

| オプション | 含まれるパッケージ                 | 用途                                                      |
| ---------- | ---------------------------------- | --------------------------------------------------------- |
| `jupyter`  | jupyter, ipykernel, ipympl, plotly | Jupyter Notebook での実行、**手動領域選択**、**3Dビュー** |
| `gui`      | PyQt6                              | スタンドアロンGUIアプリ                                   |
| `triposr`  | trimesh, omegaconf, einops         | TripoSRによる3Dメッシュ生成（別途手動インストール必要）   |
| `all`      | 上記すべて                         | フル機能                                                  |

### TripoSR のインストール（オプション）

TripoSR を使用して単一画像から直接3Dメッシュを生成するには、以下の手順でセットアップが必要です：

#### 前提条件（Windows）

- **Visual Studio Build Tools** が必要です（C++拡張のビルドに使用）
- [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) から「C++ によるデスクトップ開発」をインストール

#### インストール手順

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

#### 使用方法

```python
from shadowbox import create_pipeline, ShadowboxSettings

settings = ShadowboxSettings()
settings.model_mode = "triposr"
pipeline = create_pipeline(settings)
result = pipeline.process(image)
```

#### 注意事項

- `rembg`（背景除去）は Python 3.11+ と依存関係が競合するため、オプション化されています
- 初回実行時にモデル（約1GB）がダウンロードされます
- GPU（CUDA）推奨ですが、CPUでも動作します（処理速度は遅くなります）

> **Note**: Jupyter Notebook で以下の機能を使用するには `jupyter` オプションが必要です:
>
> - **手動領域選択**: `ipympl` により `%matplotlib widget` バックエンドでインタラクティブな操作が可能
> - **3Dビュー**: `plotly` により `render_shadowbox()` がセル内でインタラクティブ3D表示されます

## Quick Start

### Using Jupyter Notebook

```bash
uv run jupyter lab
# Open notebooks/01_quickstart.ipynb
```

### Using Python

```python
from PIL import Image
from shadowbox import create_pipeline, ShadowboxSettings

# Load image
image = Image.open("card.png")

# Create pipeline with default settings
pipeline = create_pipeline()

# Process image (auto-detect illustration area)
result = pipeline.process(image, auto_detect=True)

# View 3D shadowbox
from shadowbox.visualization import ShadowboxRenderer
renderer = ShadowboxRenderer(ShadowboxSettings().render)
renderer.render(result.mesh)
```

### Using GUI App

```bash
uv run python -m shadowbox.gui.app
```

## How It Works

1. **Load Image**: From URL or local directory
2. **Select Region**: Auto-detect or manually draw illustration area
3. **Depth Estimation**: AI model estimates per-pixel depth
4. **Clustering**: Depth values clustered into discrete layers
5. **3D Generation**: Layers rendered as stacked 3D surfaces
6. **Interactive View**: Explore with mouse rotation/zoom

## Architecture

TCG Shadowbox Generator は2つの3D生成モードをサポートしています：

### Depth Mode (default)
画像から深度マップを推定し、クラスタリングでレイヤーに分割します。

```
Image → Depth Estimation → Clustering → Mesh Generation → 3D View
              ↓                ↓              ↓
          depth map       k layers      ShadowboxMesh
```

### TripoSR Mode
単一画像から直接3Dメッシュを生成し、深度復元でレイヤー化します。

```
Image → TripoSR → 3D Mesh → Depth Recovery → Clustering → Layer Split
                     ↓            ↓              ↓
                 trimesh     depth map       k layers
```

**共通コンポーネント**: 両モードで `LayerClusterer`（クラスタリング）と Frame/BackPanel（フレーム生成）を共有しています。

詳細なアーキテクチャ分析は `docs/architecture_analysis.md` を参照してください。

## Project Structure

```text
shadowbox-generator/
├── src/shadowbox/
│   ├── core/           # Pipeline, depth, clustering, mesh
│   ├── config/         # Settings and templates
│   ├── triposr/        # TripoSR integration (generator, depth recovery, mesh splitter)
│   ├── visualization/  # 2D/3D rendering
│   ├── detection/      # Auto region detection
│   └── gui/            # GUI components
├── notebooks/          # Jupyter notebooks
├── data/templates/     # Card templates (YAML)
├── docs/               # Architecture documentation
└── tests/              # Test suite
```

## Swapping Depth Models

The depth estimator is pluggable. Currently supported:

- **Depth Anything v2** (default): Apache 2.0, lightweight, high accuracy
- **MiDaS**: MIT license, mature ecosystem
- **ZoeDepth**: MIT license, metric depth support

```python
from shadowbox.config import ShadowboxSettings

settings = ShadowboxSettings()
settings.depth.model_type = "midas"  # Switch model
pipeline = create_pipeline(settings)
```

## Development

```bash
# Run tests
uv run pytest

# Run linter
uv run ruff check src/

# Run type checker
uv run mypy src/
```

## License

MIT License
