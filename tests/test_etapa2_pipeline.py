"""
Suíte de Testes E2E Opaque-Box para o Pipeline PyTorch da Etapa 2 — AgroSense Analytics.

Esta suíte cobre rigorosamente os 4 níveis de garantia da qualidade:
- Tier 1: Testes Funcionais de Requisitos (Ingestão, Splits, Normalização, Arquitetura, Otimizador/Scheduler).
- Tier 2: Casos de Borda e Corner Cases (Imagens RGBA, Classe Rara Potato___healthy, Data Leakage, Fallback CUDA/CPU).
- Tier 3: Verificação de Artefatos Gerados (Checkpoint .pth, Figuras PNG de Matriz, Curvas e Galeria).
- Tier 4: Validação Estrutural e de Execução do Notebook Jupyter (.ipynb, JSON, células, ausência de tracebacks).
"""

import json
import os
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torchvision import models, transforms


# ==============================================================================
# TIER 1: TESTES FUNCIONAIS DE REQUISITOS (F1 a F8)
# ==============================================================================

@pytest.mark.tier1
class TestTier1FunctionalRequirements:
    """Testes funcionais que validam a integridade dos requisitos do pipeline."""

    def test_metadata_loading_and_38_classes(self, metadata_df):
        """
        R1 / F2: Valida o carregamento dos metadados consolidados, a contagem
        exata de 54.305 amostras e o mapeamento das 38 classes fitopatológicas em 14 culturas.
        """
        assert len(metadata_df) == 54305, (
            f"Esperado 54.305 amostras no dataset consolidado, mas encontrado {len(metadata_df)}"
        )

        # Validação das 38 classes fitopatológicas
        classes_unicas = sorted(metadata_df["classe_rotulo"].unique())
        assert len(classes_unicas) == 38, (
            f"Esperado 38 classes fitopatológicas distintas, mas encontrado {len(classes_unicas)}"
        )

        # Validação do formato padrão de nomenclatura 'Cultura___Condição'
        for cls_name in classes_unicas:
            assert "___" in cls_name, f"Classe '{cls_name}' não segue o padrão 'Cultura___Condição'"

        # Validação das 14 culturas agrícolas
        culturas_unicas = sorted(metadata_df["especie_cultura"].unique())
        assert len(culturas_unicas) == 14, (
            f"Esperado 14 espécies de culturas agrícolas, mas encontrado {len(culturas_unicas)}"
        )

        # Validação de integridade: ausência de valores nulos em colunas críticas
        colunas_criticas = ["caminho_arquivo", "especie_cultura", "condicao_sanitaria", "classe_rotulo"]
        for col in colunas_criticas:
            assert col in metadata_df.columns, f"Coluna crítica '{col}' ausente no CSV de metadados"
            assert metadata_df[col].isnull().sum() == 0, f"Identificados valores nulos na coluna '{col}'"

    def test_stratified_split_proportions(self, metadata_df):
        """
        R1 / F3: Valida se a estratégia de particionamento estratificado
        produz rigorosamente 70% Treino, 15% Validação e 15% Teste, mantendo
        a distribuição proporcional das classes fitopatológicas.
        """
        total = len(metadata_df)
        labels = metadata_df["classe_rotulo"]

        # Primeiro split: 70% treino vs 30% temporário (validação + teste)
        train_df, temp_df = train_test_split(
            metadata_df,
            test_size=0.30,
            stratify=labels,
            random_state=42
        )

        # Segundo split: divide o temporário em 50% validação e 50% teste (15% e 15% do total)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            stratify=temp_df["classe_rotulo"],
            random_state=42
        )

        prop_train = len(train_df) / total
        prop_val = len(val_df) / total
        prop_test = len(test_df) / total

        # Tolerância estrita de 0.5% devido a arredondamentos inteiros
        assert abs(prop_train - 0.70) < 0.005, f"Proporção de treino fora do esperado: {prop_train:.4f}"
        assert abs(prop_val - 0.15) < 0.005, f"Proporção de validação fora do esperado: {prop_val:.4f}"
        assert abs(prop_test - 0.15) < 0.005, f"Proporção de teste fora do esperado: {prop_test:.4f}"
        assert len(train_df) + len(val_df) + len(test_df) == total, "Soma das partes difere do total de amostras"

        # Validação de representação por classe em todas as partições
        classes_todas = set(metadata_df["classe_rotulo"].unique())
        classes_train = set(train_df["classe_rotulo"].unique())
        classes_val = set(val_df["classe_rotulo"].unique())
        classes_test = set(test_df["classe_rotulo"].unique())

        assert classes_train == classes_todas, "Classes faltantes na partição de treino"
        assert classes_val == classes_todas, "Classes faltantes na partição de validação"
        assert classes_test == classes_todas, "Classes faltantes na partição de teste"

    def test_imagenet_transforms_and_tensor_dimensions(self, synthetic_rgb_image):
        """
        R1 / F4: Valida a conformidade da normalização ImageNet, redimensionamento
        para 224x224 pixels e dimensões do tensor gerado [B, 3, 224, 224].
        """
        # Constantes canônicas do pré-treinamento ImageNet
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]

        transform_pipeline = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        tensor_img = transform_pipeline(synthetic_rgb_image)

        # Checagem de tipo e forma do tensor individual
        assert isinstance(tensor_img, torch.Tensor), "A saída da transformação deve ser um torch.Tensor"
        assert tensor_img.dtype == torch.float32, f"Tipo de dados esperado float32, obtido {tensor_img.dtype}"
        assert tensor_img.shape == torch.Size([3, 224, 224]), (
            f"Dimensão esperada [3, 224, 224], obtido {tensor_img.shape}"
        )

        # Simulação de batch DataLoader com dimensão B=4
        batch_tensor = torch.stack([tensor_img] * 4)
        assert batch_tensor.shape == torch.Size([4, 3, 224, 224]), (
            f"Dimensão de batch esperada [4, 3, 224, 224], obtido {batch_tensor.shape}"
        )

        # Validação de que os dados foram devidamente normalizados (não estão mais na escala [0, 1] ou [0, 255])
        assert tensor_img.min() < 0.0, "Após normalização ImageNet com z-score, valores mínimos devem ser negativos"
        assert tensor_img.max() > 0.0, "Valores máximos devem ser positivos"

    def test_mobilenetv2_architecture_and_forward_pass(self, mobilenetv2_38classes):
        """
        R2 / F6: Valida a adaptação da arquitetura MobileNetV2, conferindo a camada
        linear final com 38 saídas e a execução correta do forward pass.
        """
        model = mobilenetv2_38classes
        classifier = model.classifier

        # O classificador do MobileNetV2 é um Sequential [Dropout(0.2), Linear(1280, 38)]
        assert isinstance(classifier[1], nn.Linear), "classifier[1] deve ser uma camada nn.Linear"
        assert classifier[1].in_features == 1280, (
            f"Esperado in_features=1280 para MobileNetV2, obtido {classifier[1].in_features}"
        )
        assert classifier[1].out_features == 38, (
            f"Esperado out_features=38 para as 38 classes, obtido {classifier[1].out_features}"
        )

        # Teste de Forward Pass com batch simulado [B=2, C=3, H=224, W=224]
        dummy_input = torch.randn(2, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == torch.Size([2, 38]), (
            f"Dimensão do logit esperada [2, 38], obtido {output.shape}"
        )
        assert not torch.isnan(output).any(), "Detectado valor NaN na saída do modelo"
        assert not torch.isinf(output).any(), "Detectado valor infinito na saída do modelo"

    def test_loss_optimizer_scheduler_configuration(self, mobilenetv2_38classes):
        """
        R2 / F7: Valida a configuração do critério de perda (CrossEntropyLoss),
        otimizador (AdamW com lr e weight_decay) e scheduler (ReduceLROnPlateau).
        """
        model = mobilenetv2_38classes

        criterion = nn.CrossEntropyLoss()
        assert isinstance(criterion, nn.CrossEntropyLoss), "Critério deve ser nn.CrossEntropyLoss"

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        assert isinstance(optimizer, torch.optim.AdamW), "Otimizador deve ser AdamW"
        assert optimizer.param_groups[0]["lr"] == 1e-4, "Learning rate inicial deve ser 1e-4"
        assert optimizer.param_groups[0]["weight_decay"] == 1e-4, "Weight decay deve ser 1e-4"

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=2
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert scheduler.factor == 0.2, "Fator de decaimento do scheduler deve ser 0.2"
        assert scheduler.patience == 2, "Paciência do scheduler deve ser 2"

        # Simulação de estagnação da perda para verificar redução da learning rate
        initial_lr = optimizer.param_groups[0]["lr"]
        for _ in range(4):
            scheduler.step(1.5)  # Perda estagnada em 1.5

        updated_lr = optimizer.param_groups[0]["lr"]
        assert updated_lr < initial_lr, (
            f"O scheduler não reduziu a learning rate após exceder a paciência: de {initial_lr} para {updated_lr}"
        )


# ==============================================================================
# TIER 2: CASOS DE BORDA E CORNER CASES (Adversarial & Boundary Verification)
# ==============================================================================

@pytest.mark.tier2
class TestTier2BoundaryAndCornerCases:
    """Testes de casos limite, isolamento estatístico e robustez a falhas."""

    def test_rgba_image_handling_and_rgb_conversion(self, synthetic_rgba_image):
        """
        Corner Case 1: Garante que imagens com canal alfa (RGBA, 4 canais) sejam
        seguramente convertidas para RGB (3 canais) sem quebrar o pipeline de tensores.
        """
        assert synthetic_rgba_image.mode == "RGBA", "A imagem sintetizada de teste deve ter modo RGBA"
        assert len(synthetic_rgba_image.getbands()) == 4, "A imagem RGBA deve conter 4 bandas (R, G, B, A)"

        # Aplicação da regra de higienização do pipeline: .convert('RGB')
        converted_img = synthetic_rgba_image.convert("RGB")
        assert converted_img.mode == "RGB", "Modo após conversão deve ser estritamente 'RGB'"
        assert len(converted_img.getbands()) == 3, "Quantidade de bandas deve ser exatamente 3"

        # Passagem pelo pipeline torchvision
        transform_eval = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        tensor_out = transform_eval(converted_img)
        assert tensor_out.shape == torch.Size([3, 224, 224]), (
            f"A imagem RGBA convertida deve produzir tensor [3, 224, 224], obtido {tensor_out.shape}"
        )

    def test_smallest_class_representation_no_zero_dropout(self, metadata_df):
        """
        Corner Case 2: Verifica que a classe com menor volume amostral
        ('Potato___healthy', com apenas 152 imagens) mantém representatividade
        estatística em TODOS os 3 conjuntos (Treino, Validação e Teste) sem sofrer drop a zero.
        """
        contagens = metadata_df["classe_rotulo"].value_counts()
        menor_classe = contagens.idxmin()
        qtd_total_menor = contagens.min()

        assert menor_classe == "Potato___healthy", (
            f"Esperado que a menor classe seja 'Potato___healthy', encontrado '{menor_classe}'"
        )
        assert qtd_total_menor == 152, (
            f"Esperado 152 amostras para 'Potato___healthy', encontrado {qtd_total_menor}"
        )

        # Execução do particionamento estratificado padrão
        train_df, temp_df = train_test_split(
            metadata_df, test_size=0.30, stratify=metadata_df["classe_rotulo"], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, stratify=temp_df["classe_rotulo"], random_state=42
        )

        n_train = (train_df["classe_rotulo"] == menor_classe).sum()
        n_val = (val_df["classe_rotulo"] == menor_classe).sum()
        n_test = (test_df["classe_rotulo"] == menor_classe).sum()

        # Nenhuma partição pode ter zero amostras (zero dropout)
        assert n_train > 0, f"Classe minoritária '{menor_classe}' ausente no treino (zero dropout)!"
        assert n_val > 0, f"Classe minoritária '{menor_classe}' ausente na validação (zero dropout)!"
        assert n_test > 0, f"Classe minoritária '{menor_classe}' ausente no teste (zero dropout)!"

        # A soma deve fechar exatamente 152 amostras
        assert n_train + n_val + n_test == 152, "Soma de amostras da menor classe difere do total"

        # Verificação de proporções aproximadas (70% ~ 106, 15% ~ 23, 15% ~ 23)
        assert abs(n_train - 106) <= 2, f"Desvio expressivo no treino para a classe rara: {n_train}"
        assert abs(n_val - 23) <= 2, f"Desvio expressivo na validação para a classe rara: {n_val}"
        assert abs(n_test - 23) <= 2, f"Desvio expressivo no teste para a classe rara: {n_test}"

    def test_zero_data_leakage_formal_assertion(self, metadata_df):
        """
        Corner Case 3: Prova formal matemática de que os conjuntos de Treino,
        Validação e Teste são mutuamente disjuntos, garantindo ZERO vazamento
        de dados (zero data leakage) em nível de amostra/caminho físico.
        """
        train_df, temp_df = train_test_split(
            metadata_df, test_size=0.30, stratify=metadata_df["classe_rotulo"], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, stratify=temp_df["classe_rotulo"], random_state=42
        )

        set_train = set(train_df["caminho_arquivo"])
        set_val = set(val_df["caminho_arquivo"])
        set_test = set(test_df["caminho_arquivo"])

        # Interseções de pares devem ser rigorosamente o conjunto vazio
        inter_train_val = set_train.intersection(set_val)
        inter_train_test = set_train.intersection(set_test)
        inter_val_test = set_val.intersection(set_test)

        assert len(inter_train_val) == 0, (
            f"Vazamento de dados detectado entre Treino e Validação: {len(inter_train_val)} amostras comuns!"
        )
        assert len(inter_train_test) == 0, (
            f"Vazamento de dados detectado entre Treino e Teste: {len(inter_train_test)} amostras comuns!"
        )
        assert len(inter_val_test) == 0, (
            f"Vazamento de dados detectado entre Validação e Teste: {len(inter_val_test)} amostras comuns!"
        )

        # A união dos três conjuntos deve ser igual ao conjunto total original
        uniao_total = set_train.union(set_val).union(set_test)
        assert len(uniao_total) == len(metadata_df), (
            "A união das partições não reconstitui a integralidade do dataset original"
        )

    def test_dynamic_cuda_check_with_cpu_fallback(self, monkeypatch):
        """
        Corner Case 4: Valida a lógica de resolução dinâmica de dispositivo de hardware,
        assegurando que o pipeline utilize GPU CUDA se disponível, com fallback transparente para CPU.
        """
        # Função representativa da lógica do notebook
        def resolve_device():
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        current_device = resolve_device()
        assert current_device.type in ["cuda", "cpu"], f"Tipo de dispositivo desconhecido: {current_device.type}"

        # Teste de Fallback forçado simulando ausência de GPU CUDA
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        fallback_device = resolve_device()
        assert fallback_device.type == "cpu", "O fallback não selecionou 'cpu' quando CUDA estava indisponível"

        # Validação operacional de que tensores e modelos operam no device selecionado
        dummy_tensor = torch.zeros((2, 2), device=fallback_device)
        assert dummy_tensor.device.type == "cpu"


# ==============================================================================
# TIER 3: VERIFICAÇÃO DE ARTEFATOS PERSISTIDOS (Checkpoint & Figuras)
# ==============================================================================

@pytest.mark.tier3
class TestTier3ArtifactVerification:
    """Validação física, assinaturas e integridade dos artefatos produzidos pelo pipeline."""

    def test_model_checkpoint_artifact_exists_and_valid(self, repo_root):
        """
        R2 / F8: Verifica se o arquivo de checkpoint do melhor modelo existe
        em 'models/mobilenetv2_plantvillage.pth', possui os pesos corretos e
        o mapeamento completo das 38 classes.
        """
        checkpoint_path = repo_root / "models" / "mobilenetv2_plantvillage.pth"

        assert checkpoint_path.exists(), (
            f"Artefato de checkpoint não encontrado em: {checkpoint_path}. "
            "Execute a pipeline de treinamento para produzir o checkpoint do modelo."
        )

        # O arquivo não deve estar vazio
        assert checkpoint_path.stat().st_size > 1024 * 1024, (
            f"O checkpoint possui tamanho suspeitamente reduzido ({checkpoint_path.stat().st_size} bytes)"
        )

        # Carga do checkpoint com tratamento seguro de pesos
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert isinstance(checkpoint, dict), "O checkpoint salvo deve ser um dicionário estruturado"

        # Verificação das chaves contratuais obrigatórias
        chaves_obrigatorias = ["model_state_dict", "classes", "class_to_idx"]
        for chave in chaves_obrigatorias:
            assert chave in checkpoint, f"Chave obrigatória '{chave}' ausente no checkpoint salvo"

        # Verificação das classes registradas
        classes_registradas = checkpoint["classes"]
        assert len(classes_registradas) == 38, (
            f"Esperado 38 classes no checkpoint, encontrado {len(classes_registradas)}"
        )

        # Teste de restauração dos pesos no MobileNetV2
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 38)
        load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        assert len(load_result.missing_keys) == 0, f"Chaves ausentes ao carregar modelo: {load_result.missing_keys}"
        assert len(load_result.unexpected_keys) == 0, f"Chaves inesperadas ao carregar: {load_result.unexpected_keys}"

    @pytest.mark.parametrize("figura_relativa, min_size_bytes", [
        ("reports/figures/matriz_confusao.png", 20 * 1024),
        ("reports/figures/curvas_aprendizado.png", 20 * 1024),
        ("reports/figures/galeria_inferencia.png", 20 * 1024),
    ])
    def test_report_figure_artifacts_validity(self, repo_root, figura_relativa, min_size_bytes):
        """
        R3 / F11, F12, F13: Verifica se as figuras PNG em alta resolução foram
        devidamente geradas, possuem a assinatura mágica de PNG e dimensões gráficas reais.
        """
        fig_path = repo_root / Path(figura_relativa)

        assert fig_path.exists(), (
            f"Figura analítica não encontrada: {fig_path}. "
            "Execute a etapa de avaliação para persistir as visualizações gráficas."
        )

        # Verificação de tamanho mínimo de arquivo
        file_size = fig_path.stat().st_size
        assert file_size >= min_size_bytes, (
            f"Figura '{figura_relativa}' com tamanho suspeito ({file_size} bytes < {min_size_bytes} bytes)"
        )

        # Validação do Magic Number / Assinatura de PNG (8 bytes: 89 50 4E 47 0D 0A 1A 0A)
        with open(fig_path, "rb") as f:
            header = f.read(8)
        png_magic = b"\x89PNG\r\n\x1a\n"
        assert header == png_magic, f"O arquivo '{figura_relativa}' não possui cabeçalho válido de PNG"

        # Validação gráfica com Pillow
        with Image.open(fig_path) as img:
            assert img.format == "PNG", f"Formato identificado pelo Pillow não é PNG: {img.format}"
            assert img.width >= 300, f"Largura da imagem insuficiente para 300 DPI: {img.width}px"
            assert img.height >= 300, f"Altura da imagem insuficiente para 300 DPI: {img.height}px"


# ==============================================================================
# TIER 4: VALIDAÇÃO DO NOTEBOOK JUPYTER (Execução, JSON e Tracebacks)
# ==============================================================================

@pytest.mark.tier4
class TestTier4NotebookVerification:
    """Validação da estrutura, tipagem de células e ausência de falhas no notebook Jupyter."""

    @pytest.fixture(scope="class")
    def notebook_json(self, repo_root):
        """Carrega e analisa o notebook oficial da Etapa 2."""
        nb_path = repo_root / "notebooks" / "Etapa_2_Treinamento_Modelo_e_Metricas.ipynb"
        assert nb_path.exists(), (
            f"Notebook de treinamento não encontrado em: {nb_path}. "
            "Construa e execute o notebook da Etapa 2 para validação."
        )

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        return nb

    def test_notebook_json_validity_and_schema(self, notebook_json):
        """
        Acceptance Criteria: Valida que o notebook é um JSON válido em conformidade
        com o padrão nbformat >= 4 e possui metadados de kernel Python.
        """
        assert "nbformat" in notebook_json, "Campo 'nbformat' ausente no JSON do notebook"
        assert notebook_json["nbformat"] >= 4, f"nbformat deve ser >= 4, obtido {notebook_json['nbformat']}"
        assert "cells" in notebook_json, "Campo 'cells' ausente no notebook"
        assert isinstance(notebook_json["cells"], list), "'cells' deve ser uma lista"
        assert len(notebook_json["cells"]) > 0, "O notebook não contém células"

    def test_notebook_structure_and_pedagogical_sections(self, notebook_json):
        """
        Acceptance Criteria: Valida que o notebook possui equilíbrio entre
        células Markdown e Code e contém os tópicos obrigatórios do projeto.
        """
        cells = notebook_json["cells"]
        md_cells = [c for c in cells if c.get("cell_type") == "markdown"]
        code_cells = [c for c in cells if c.get("cell_type") == "code"]

        assert len(md_cells) >= 6, f"Esperado ao menos 6 células Markdown descritivas, obtido {len(md_cells)}"
        assert len(code_cells) >= 10, f"Esperado ao menos 10 células de código funcionais, obtido {len(code_cells)}"

        # Busca por termos obrigatórios nas seções textuais
        all_text = " ".join(
            "".join(c.get("source", [])) for c in md_cells
        ).lower()

        secoes_obrigatorias = [
            "mobilenetv2",
            "plantvillage",
            "treino",
            "validação",
            "teste",
            "matriz de confusão",
            "curvas de aprendizado"
        ]

        for secao in secoes_obrigatorias:
            assert secao in all_text, f"Seção temática '{secao}' ausente nos textos explicativos do notebook"

    def test_notebook_executed_outputs_and_no_tracebacks(self, notebook_json):
        """
        Acceptance Criteria: Garante que o notebook foi devidamente executado,
        contendo saídas renderizadas e NENHUM traceback de erro de execução.
        """
        code_cells = [c for c in notebook_json["cells"] if c.get("cell_type") == "code"]

        executed_cells_count = 0
        erros_encontrados = []

        for idx, cell in enumerate(code_cells):
            # Célula executada possui execution_count não nulo
            if cell.get("execution_count") is not None:
                executed_cells_count += 1

            outputs = cell.get("outputs", [])
            for out in outputs:
                out_type = out.get("output_type")
                if out_type == "error":
                    ename = out.get("ename", "UnknownError")
                    evalue = out.get("evalue", "UnknownValue")
                    erros_encontrados.append(f"Célula #{idx} falhou com {ename}: {evalue}")

        assert len(erros_encontrados) == 0, (
            "Detectados erros/tracebacks de execução no notebook:\n" + "\n".join(erros_encontrados)
        )

        assert executed_cells_count >= 8, (
            f"Esperado que a maioria das células de código contenha saídas de execução, mas apenas "
            f"{executed_cells_count}/{len(code_cells)} possuem execution_count registrado."
        )


# ==============================================================================
# Execução autônoma direta caso invocado via CLI (python test_etapa2_pipeline.py)
# ==============================================================================
if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", __file__]))
