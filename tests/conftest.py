"""
Fixtures e configurações compartilhadas para a suíte de testes do AgroSense Analytics (Etapa 2).
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models

# Adiciona a raiz do projeto ao sys.path para importação de módulos se necessário
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Retorna o caminho absoluto para a raiz do repositório."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def metadata_path(repo_root) -> Path:
    """Retorna o caminho para o arquivo CSV de metadados consolidado."""
    candidate = repo_root / "data" / "metadados_plantvillage.csv"
    return candidate


@pytest.fixture(scope="session")
def metadata_df(metadata_path) -> pd.DataFrame:
    """Carrega o DataFrame de metadados do dataset PlantVillage."""
    assert metadata_path.exists(), f"Arquivo de metadados não encontrado: {metadata_path}"
    df = pd.read_csv(metadata_path)
    return df


@pytest.fixture(scope="session")
def dataset_color_dir(repo_root) -> Path:
    """Localiza o diretório raiz das imagens coloridas do PlantVillage."""
    candidates = [
        repo_root.parent / "plantvillage dataset" / "color",
        repo_root / "plantvillage dataset" / "color",
        repo_root.parent / "plantvillage" / "color"
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    # Retorna o primeiro caminho esperado mesmo se não existir
    return candidates[0]


@pytest.fixture
def synthetic_rgb_image() -> Image.Image:
    """Gera uma imagem sintética RGB de 256x256 para testes de transformação."""
    return Image.new("RGB", (256, 256), color=(110, 140, 60))


@pytest.fixture
def synthetic_rgba_image() -> Image.Image:
    """Gera uma imagem sintética RGBA de 256x256 para teste de corner case de 4 canais."""
    return Image.new("RGBA", (256, 256), color=(110, 140, 60, 200))


@pytest.fixture
def mobilenetv2_38classes() -> nn.Module:
    """Instancia um modelo MobileNetV2 adaptado com 38 classes de saída."""
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 38)
    return model
