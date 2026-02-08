# TCG Shadowbox Generator

[日本語版はこちら (Japanese)](README_ja.md)

Transform TCG card illustrations into interactive 3D shadowbox displays using AI-powered depth estimation.

## Features

- **Depth Estimation**: Uses Depth Anything v2 (lightweight, Apache 2.0 license)
- **Auto/Manual Region Selection**: 10 ensemble detection methods with confidence scoring, or manual selection
- **Layer Clustering**: Automatically finds optimal number of layers using silhouette analysis
- **Contour Cut Mode**: Depth-based contour masking for natural layer shapes (front layers narrow, back layers wide)
- **Mesh Export**: Export to STL, OBJ, and PLY formats for 3D printing or external tools
- **GUI App (PyQt6)**: Region selection, parameter tuning, 3D preview, and mesh export with bilingual UI (English / Japanese)
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

| Option    | Packages                           | Purpose                                              |
| --------- | ---------------------------------- | ---------------------------------------------------- |
| `jupyter` | jupyter, ipykernel, ipympl, plotly | Jupyter Notebook, **manual region selection**, **3D view** |
| `gui`     | PyQt6                              | Standalone GUI app                                   |
| `triposr` | trimesh, rtree, omegaconf, einops, pyrender, rembg, onnxruntime, pymatting | TripoSR 3D mesh generation (manual setup required) |
| `all`     | All of the above                   | Full features                                        |

### TripoSR Installation (Optional)

To generate 3D meshes directly from a single image using TripoSR, follow these setup steps:

#### Prerequisites (Windows)

- **Visual Studio Build Tools** required (for building C++ extensions)
- Install "Desktop development with C++" from [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

#### Setup Steps

```bash
# 1. Install dependencies
uv sync --extra triposr

# 2. Clone TripoSR (place in project root)
git clone https://github.com/VAST-AI-Research/TripoSR.git

# 3. Build and install torchmcubes (C++ extension)
uv pip install scikit-build-core cmake ninja pybind11

# On Windows, set CMAKE_PREFIX_PATH before building
set CMAKE_PREFIX_PATH=.venv\Lib\site-packages\torch\share\cmake
uv pip install git+https://github.com/tatsy/torchmcubes.git --no-build-isolation
```

> **Note**: The `TripoSR/` directory is auto-detected when placed in the project root. No manual PYTHONPATH configuration is needed.

#### Usage

```python
from shadowbox import create_pipeline, ShadowboxSettings

settings = ShadowboxSettings()
settings.model_mode = "triposr"
pipeline = create_pipeline(settings)
result = pipeline.process(image)
```

#### Notes

- `rembg` (background removal) is included in the `triposr` extra
- The model (~1GB) is downloaded on first run
- GPU (CUDA) recommended, but CPU is also supported (slower)

> **Note**: The `jupyter` extra is required for these features in Jupyter Notebook:
>
> - **Manual region selection**: Interactive operation via `ipympl` with `%matplotlib widget` backend
> - **3D view**: Interactive 3D display in cells via `plotly` with `render_shadowbox()`

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
5. **Mask Mode**: Choose between cluster labels (flat layers) or contour cut (depth-based natural shapes)
6. **3D Generation**: Layers rendered as stacked 3D surfaces
7. **Interactive View**: Explore with mouse rotation/zoom

## Architecture

TCG Shadowbox Generator supports two 3D generation modes:

### Depth Mode (default)
Estimates a depth map from the image and splits it into layers via clustering.

```
Image → Depth Estimation → DepthToMeshProcessor → ShadowboxMesh → 3D View
              ↓                    ↓
          depth map      clustering + mesh generation
```

### TripoSR Mode
Generates a 3D mesh directly from a single image, then recovers depth for layer splitting.

```
Image → TripoSR → 3D Mesh → Depth Recovery → DepthToMeshProcessor → ShadowboxMesh
                     ↓            ↓                    ↓
                 trimesh     depth map       clustering + mesh generation
```

**Shared Components**: Both modes converge on `DepthToMeshProcessor`, which handles clustering (`LayerClusterer`) and mesh generation. Frame/BackPanel factories are also shared.

See `docs/architecture_analysis.md` for detailed architecture analysis.

## Project Structure

```text
shadowbox-generator/
├── src/shadowbox/
│   ├── core/           # Pipeline, depth, clustering, mesh
│   ├── config/         # Settings and templates
│   ├── triposr/        # TripoSR integration (generator, depth recovery, mesh splitter)
│   ├── visualization/  # 2D/3D rendering
│   ├── detection/      # Auto region detection (10 ensemble methods)
│   ├── gui/            # PyQt6 GUI app with widget tabs
│   └── utils/          # Image loading and processing utilities
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
