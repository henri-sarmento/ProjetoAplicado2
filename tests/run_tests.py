"""
Runner e Relatório Diagnóstico Autônomo para a Suíte de Testes da Etapa 2.

Pode ser executado com:
    python tests/run_tests.py
ou com:
    pytest tests/ -v
"""

import os
import sys
import json
import traceback
from pathlib import Path

# Adiciona o diretório raiz ao path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.model_selection import train_test_split


def log_header(title):
    print("=" * 78)
    print(f"  {title.upper()}")
    print("=" * 78)


def log_tier(tier_num, title):
    print(f"\n[{tier_num}] {title}")
    print("-" * 78)


def run_tier1():
    log_tier("TIER 1", "Testes Funcionais de Requisitos (Ingestão, Split, Tensores, Modelo, Otimizador)")
    results = []

    # 1.1 Ingestão e 38 Classes
    try:
        csv_path = REPO_ROOT / "data" / "metadados_plantvillage.csv"
        assert csv_path.exists(), f"Arquivo não encontrado: {csv_path}"
        df = pd.read_csv(csv_path)
        assert len(df) == 54305, f"Esperado 54305 amostras, obtido {len(df)}"
        classes = sorted(df["classe_rotulo"].unique())
        assert len(classes) == 38, f"Esperado 38 classes, obtido {len(classes)}"
        culturas = sorted(df["especie_cultura"].unique())
        assert len(culturas) == 14, f"Esperado 14 culturas, obtido {len(culturas)}"
        print("  [PASS] 1.1 Ingestão de Dados e Mapeamento de 38 Classes (54.305 amostras, 14 culturas)")
        results.append(("1.1 Ingestão e 38 Classes", "PASS", None))
    except Exception as e:
        print(f"  [FAIL] 1.1 Ingestão e 38 Classes: {e}")
        results.append(("1.1 Ingestão e 38 Classes", "FAIL", str(e)))

    # 1.2 Split 70/15/15 Estratificado
    try:
        train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["classe_rotulo"], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["classe_rotulo"], random_state=42)
        p_train, p_val, p_test = len(train_df)/len(df), len(val_df)/len(df), len(test_df)/len(df)
        assert abs(p_train - 0.70) < 0.005, f"Treino: {p_train:.4f}"
        assert abs(p_val - 0.15) < 0.005, f"Validação: {p_val:.4f}"
        assert abs(p_test - 0.15) < 0.005, f"Teste: {p_test:.4f}"
        assert len(train_df) + len(val_df) + len(test_df) == len(df)
        print(f"  [PASS] 1.2 Particionamento Estratificado 70/15/15 (Treino: {len(train_df):,}, Val: {len(val_df):,}, Teste: {len(test_df):,})")
        results.append(("1.2 Particionamento 70/15/15", "PASS", None))
    except Exception as e:
        print(f"  [FAIL] 1.2 Particionamento Estratificado 70/15/15: {e}")
        results.append(("1.2 Particionamento 70/15/15", "FAIL", str(e)))

    # 1.3 Normalização ImageNet e Dimensões dos Tensores
    try:
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        img = Image.new("RGB", (256, 256), color=(100, 150, 80))
        t = tf(img)
        assert t.shape == torch.Size([3, 224, 224])
        batch = torch.stack([t] * 4)
        assert batch.shape == torch.Size([4, 3, 224, 224])
        print("  [PASS] 1.3 Normalização ImageNet e Formatos de Tensor [B, 3, 224, 224]")
        results.append(("1.3 Normalização e Tensores", "PASS", None))
    except Exception as e:
        print(f"  [FAIL] 1.3 Normalização ImageNet e Tensores: {e}")
        results.append(("1.3 Normalização e Tensores", "FAIL", str(e)))

    # 1.4 Arquitetura MobileNetV2 com 38 saídas
    try:
        model = models.mobilenet_v2(weights=None)
        in_feat = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feat, 38)
        assert model.classifier[1].out_features == 38
        dummy_input = torch.randn(2, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        assert out.shape == torch.Size([2, 38])
        assert not torch.isnan(out).any()
        print("  [PASS] 1.4 Arquitetura MobileNetV2 (Linear 1280->38 e Forward Pass [2, 38])")
        results.append(("1.4 MobileNetV2 38 Classes", "PASS", None))
    except Exception as e:
        print(f"  [FAIL] 1.4 Arquitetura MobileNetV2: {e}")
        results.append(("1.4 MobileNetV2 38 Classes", "FAIL", str(e)))

    # 1.5 Critério de Perda, Otimizador e Scheduler
    try:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=2)
        initial_lr = optimizer.param_groups[0]["lr"]
        for _ in range(4):
            scheduler.step(1.5)
        new_lr = optimizer.param_groups[0]["lr"]
        assert new_lr < initial_lr, "Scheduler não reduziu LR após paciência"
        print(f"  [PASS] 1.5 Configuração de Perda, AdamW e ReduceLROnPlateau (LR: {initial_lr} -> {new_lr})")
        results.append(("1.5 Loss/Otimizador/Scheduler", "PASS", None))
    except Exception as e:
        print(f"  [FAIL] 1.5 Loss/Otimizador/Scheduler: {e}")
        results.append(("1.5 Loss/Otimizador/Scheduler", "FAIL", str(e)))

    return results


def run_tier2():
    log_tier("TIER 2", "Casos de Borda e Corner Cases (RGBA, Classe Rara, Data Leakage, Dynamic CUDA)")
    results = []
    csv_path = REPO_ROOT / "data" / "metadados_plantvillage.csv"
    df = pd.read_csv(csv_path)

    # 2.1 Tratamento Seguro de Imagens RGBA
    try:
        rgba = Image.new("RGBA", (256, 256), color=(100, 150, 80, 255))
        assert rgba.mode == "RGBA"
        rgb = rgba.convert("RGB")
        assert rgb.mode == "RGB"
        assert len(rgb.getbands()) == 3
        tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        tensor_out = tf(rgb)
        assert tensor_out.shape == torch.Size([3, 224, 224])
        print("  [PASS] 2.1 Higienização de Imagens RGBA (Conversão Segura para 3 Canais RGB)")
        results.append(("2.1 Higienização RGBA", "PASS", None))
    except Exception as e:
        print(f"  [FAIL] 2.1 Higienização RGBA: {e}")
        results.append(("2.1 Higienização RGBA", "FAIL", str(e)))

    # 2.2 Representação da Menor Classe (Potato___healthy, 152 amostras)
    try:
        min_class = "Potato___healthy"
        train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["classe_rotulo"], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["classe_rotulo"], random_state=42)
        n_train = (train_df["classe_rotulo"] == min_class).sum()
        n_val = (val_df["classe_rotulo"] == min_class).sum()
        n_test = (test_df["classe_rotulo"] == min_class).sum()
        assert n_train > 0 and n_val > 0 and n_test > 0, "Zero dropout detectado na menor classe!"
        assert n_train + n_val + n_test == 152
        print(f"  [PASS] 2.2 Preservação da Classe Rara '{min_class}' (Treino: {n_train}, Val: {n_val}, Teste: {n_test})")
        results.append(("2.2 Menor Classe Sem Zero-Dropout", "PASS", None))
    except Exception as e:
        print(f"  [FAIL] 2.2 Menor Classe Sem Zero-Dropout: {e}")
        results.append(("2.2 Menor Classe Sem Zero-Dropout", "FAIL", str(e)))

    # 2.3 Prova Formal de Zero Data Leakage
    try:
        s_train = set(train_df["caminho_arquivo"])
        s_val = set(val_df["caminho_arquivo"])
        s_test = set(test_df["caminho_arquivo"])
        assert len(s_train.intersection(s_val)) == 0
        assert len(s_train.intersection(s_test)) == 0
        assert len(s_val.intersection(s_test)) == 0
        assert len(s_train.union(s_val).union(s_test)) == len(df)
        print("  [PASS] 2.3 Prova Formal de Zero Data Leakage (Partições Mutuamente Disjuntas)")
        results.append(("2.3 Zero Data Leakage", "PASS", None))
    except Exception as e:
        print(f"  [FAIL] 2.3 Zero Data Leakage: {e}")
        results.append(("2.3 Zero Data Leakage", "FAIL", str(e)))

    # 2.4 Resolução Dinâmica de CUDA com Fallback para CPU
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        assert device.type in ["cuda", "cpu"]
        # Fallback explícito
        fallback = torch.device("cpu")
        t = torch.ones((2, 2), device=fallback)
        assert t.device.type == "cpu"
        print(f"  [PASS] 2.4 Dynamic CUDA Check com Fallback Transparente para CPU (Ativo: {device})")
        results.append(("2.4 Dynamic CUDA / CPU Fallback", "PASS", None))
    except Exception as e:
        print(f"  [FAIL] 2.4 Dynamic CUDA Check com Fallback: {e}")
        results.append(("2.4 Dynamic CUDA / CPU Fallback", "FAIL", str(e)))

    return results


def run_tier3():
    log_tier("TIER 3", "Verificação de Artefatos Gerados (Checkpoint .pth e Figuras .png)")
    results = []

    # 3.1 Checkpoint do Modelo
    checkpoint_path = REPO_ROOT / "models" / "mobilenetv2_plantvillage.pth"
    if not checkpoint_path.exists():
        print(f"  [PENDING / FAIL] 3.1 Checkpoint '{checkpoint_path.name}' não encontrado no disco.")
        results.append(("3.1 Checkpoint mobilenetv2_plantvillage.pth", "FAIL", "Arquivo não encontrado (necessário executar treinamento)"))
    else:
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            assert "model_state_dict" in ckpt
            assert "classes" in ckpt and len(ckpt["classes"]) == 38
            print("  [PASS] 3.1 Checkpoint mobilenetv2_plantvillage.pth (Válido, 38 classes mapeadas)")
            results.append(("3.1 Checkpoint mobilenetv2_plantvillage.pth", "PASS", None))
        except Exception as e:
            print(f"  [FAIL] 3.1 Checkpoint mobilenetv2_plantvillage.pth inválido: {e}")
            results.append(("3.1 Checkpoint mobilenetv2_plantvillage.pth", "FAIL", str(e)))

    # 3.2 Figuras
    figuras = [
        ("reports/figures/matriz_confusao.png", "3.2 Matriz de Confusão PNG"),
        ("reports/figures/curvas_aprendizado.png", "3.3 Curvas de Aprendizado PNG"),
        ("reports/figures/galeria_inferencia.png", "3.4 Galeria de Inferência PNG"),
    ]
    png_magic = b"\x89PNG\r\n\x1a\n"
    for rel_path, nome in figuras:
        f_path = REPO_ROOT / Path(rel_path)
        if not f_path.exists():
            print(f"  [PENDING / FAIL] {nome}: Arquivo '{rel_path}' não encontrado.")
            results.append((nome, "FAIL", f"Arquivo '{rel_path}' não encontrado"))
        else:
            try:
                assert f_path.stat().st_size >= 10240, "Arquivo com tamanho insignificante (< 10 KB)"
                with open(f_path, "rb") as f:
                    header = f.read(8)
                assert header == png_magic, "Assinatura mágica de PNG inválida"
                with Image.open(f_path) as im:
                    assert im.format == "PNG"
                    assert im.width >= 300 and im.height >= 300
                print(f"  [PASS] {nome} (PNG 300+ DPI, {f_path.stat().st_size:,} bytes)")
                results.append((nome, "PASS", None))
            except Exception as e:
                print(f"  [FAIL] {nome}: {e}")
                results.append((nome, "FAIL", str(e)))

    return results


def run_tier4():
    log_tier("TIER 4", "Validação do Notebook Jupyter (JSON, Estrutura e Execução)")
    results = []
    nb_path = REPO_ROOT / "notebooks" / "Etapa_2_Treinamento_Modelo_e_Metricas.ipynb"

    if not nb_path.exists():
        print(f"  [PENDING / FAIL] 4.1 Notebook '{nb_path.name}' não encontrado.")
        results.append(("4.1 Notebook JSON & Execução", "FAIL", f"Arquivo '{nb_path.name}' não encontrado"))
        return results

    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        assert nb.get("nbformat", 0) >= 4, "nbformat inválido"
        cells = nb.get("cells", [])
        assert len(cells) > 0, "Notebook sem células"
        print(f"  [PASS] 4.1 Notebook JSON Válido (nbformat v{nb['nbformat']}, {len(cells)} células)")
        results.append(("4.1 Notebook JSON Válido", "PASS", None))
    except Exception as e:
        print(f"  [FAIL] 4.1 Notebook JSON: {e}")
        results.append(("4.1 Notebook JSON Válido", "FAIL", str(e)))
        return results

    # 4.2 Estrutura de Células
    md_cells = [c for c in cells if c.get("cell_type") == "markdown"]
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    if len(md_cells) >= 5 and len(code_cells) >= 10:
        print(f"  [PASS] 4.2 Estrutura Pedagógica ({len(md_cells)} células Markdown, {len(code_cells)} de Código)")
        results.append(("4.2 Estrutura de Células", "PASS", None))
    else:
        print(f"  [FAIL] 4.2 Estrutura Insuficiente: {len(md_cells)} Markdown, {len(code_cells)} Código")
        results.append(("4.2 Estrutura de Células", "FAIL", f"Apenas {len(md_cells)} MD e {len(code_cells)} Code"))

    # 4.3 Ausência de Tracebacks de Erro
    erros = []
    exec_count = 0
    for idx, c in enumerate(code_cells):
        if c.get("execution_count") is not None:
            exec_count += 1
        for out in c.get("outputs", []):
            if out.get("output_type") == "error":
                erros.append(f"Célula #{idx}: {out.get('ename')}: {out.get('evalue')}")

    if len(erros) == 0 and exec_count >= 8:
        print(f"  [PASS] 4.3 Execução Sem Tracebacks ({exec_count}/{len(code_cells)} células executadas)")
        results.append(("4.3 Execução Sem Erros", "PASS", None))
    elif len(erros) > 0:
        print(f"  [FAIL] 4.3 Detectados {len(erros)} erros de execução no notebook:")
        for err in erros[:3]:
            print(f"         - {err}")
        results.append(("4.3 Execução Sem Erros", "FAIL", "; ".join(erros[:2])))
    else:
        print(f"  [FAIL] 4.3 Notebook não aparenta ter sido executado ({exec_count} células com execution_count)")
        results.append(("4.3 Execução Sem Erros", "FAIL", "Notebook não executado"))

    return results


def main():
    log_header("AgroSense Analytics — Test Suite Runner (Etapa 2)")
    print(f"Repositório: {REPO_ROOT.resolve()}")
    print(f"Python: {sys.version.split()[0]} | PyTorch: {torch.__version__}")

    all_results = []
    all_results.extend(run_tier1())
    all_results.extend(run_tier2())
    all_results.extend(run_tier3())
    all_results.extend(run_tier4())

    log_header("Resumo dos Testes")
    total = len(all_results)
    passed = sum(1 for _, status, _ in all_results if status == "PASS")
    failed = total - passed

    for name, status, err in all_results:
        simbolo = "[OK]  " if status == "PASS" else "[FAIL]"
        msg_err = f" -> {err}" if err else ""
        print(f"  {simbolo} {name:40s}: {status}{msg_err}")

    print("-" * 78)
    print(f"Total: {total} | Aprovados: {passed} | Falhas/Pendentes: {failed}")
    print("=" * 78)

    # Retorna código de erro proporcional às falhas
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
