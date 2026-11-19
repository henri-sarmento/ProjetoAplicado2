# AgroSense Analytics — Classificação Fitopatológica Foliar com MobileNetV2 e Arquitetura Edge AI

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c.svg)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.21%2B-orange.svg)](https://pytorch.org/vision/)
[![Jupyter Notebooks](https://img.shields.io/badge/Notebooks-100%25%20.ipynb-F37626.svg)](Etapa%204/notebooks/)
[![Status: Etapa 4 Concluída](https://img.shields.io/badge/Status-Etapa%204%20(Entrega%20Final)-brightgreen.svg)](docs/)

> **Projeto Aplicado II — Banco de Dados — EAD**  
> **Universidade Presbiteriana Mackenzie** — Faculdade de Computação e Informática (FCI)  
> **Professor Orientador:** Prof. Anderson Adaime de Borba  
> **Equipe:** Henrique Sarmento · Felipe Magalhães de Melo Costa · Jefferson Candido Moreira · Karla Maria Ramos da Silva

---

## Acesso Rápido aos Entregáveis Finais (Etapa 4)

* **Apresentação em Vídeo (YouTube):** `[Inserir link do vídeo publicado no YouTube após gravação]`
* **Relatório Técnico Final (ABNT):** [`docs/Relatorio_Etapa4_AgroSense_ABNT.pdf`](docs/Relatorio_Etapa4_AgroSense_ABNT.pdf) | [`docs/Relatorio_Etapa4_AgroSense_ABNT.docx`](docs/Relatorio_Etapa4_AgroSense_ABNT.docx)
* **Apresentação Executiva:** [`docs/Apresentacao_Etapa4_AgroSense.pptx`](docs/Apresentacao_Etapa4_AgroSense.pptx)
* **Diretório para Replicação em Código:** [`Etapa 4/notebooks/`](Etapa%204/notebooks/) 

---

## 1. Como Navegar pelo Repositório (Por Onde Começar?)

Este repositório preserva o histórico evolutivo completo da disciplina (**Etapas 1, 2, 3 e 4**), mas **toda a replicação final e oficial do projeto deve ser realizada exclusivamente a partir da pasta [`Etapa 4/`](Etapa%204/)**:

| Pasta do Repositório | Papel na Estrutura | Deve ser usada para replicar o resultado final? |
| :--- | :--- | :---: |
| **[`Etapa 4/`](Etapa%204/)** | **Entrega Final Consolidada (Caminho Canônico):** Contém o catálogo auditado (`54.305` registros), o modelo final (`mobilenetv2_plantvillage.pth`, `26,38 MB`), as métricas e figuras de alta resolução (300 DPI) e os **4 Jupyter Notebooks (`.ipynb`)** que reproduzem toda a solução sem depender de scripts `.py` externos. | **SIM (Recomendado)** |
| **[`docs/`](docs/)** | **Biblioteca de Documentos Finais:** Reúne todos os relatórios oficiais (`.pdf` e `.docx` das Etapas 1 a 4) e os slides executivos finais (`Apresentacao_Etapa4_AgroSense.pptx`). | Consulta / Leitura |
| [`Etapa 3/`](Etapa%203/) | *Histórico Acadêmico (Etapa 3):* Versão preliminar da modelagem de negócios e storytelling antes das correções e refinamentos finais da Etapa 4. | Não (Apenas histórico) |
| [`Etapa 2/`](Etapa%202/) | *Histórico Acadêmico (Etapa 2):* Primeiros experimentos de treinamento, scripts modulares preliminares (`src/`) e testes unitários iniciais (`tests/`). | Não (Apenas histórico) |
| [`Etapa 1/`](Etapa%201/) | *Histórico Acadêmico (Etapa 1):* Concepção inicial do problema, justificativa e fundamentação teórica. | Não (Apenas histórico) |

---

## 2. Visão Geral e Delimitação de Escopo (Construído vs. Proposto)

O **AgroSense Analytics** desenvolve e valida um motor leve de Visão Computacional baseado na arquitetura **MobileNetV2** (SANDLER et al., 2018) para classificação de patologias foliares e estrutura, a partir dele, uma proposta arquitetural de aplicação *Edge AI* (**TRL 4**) e um modelo de negócios **B2B SaaS** para cooperativas agrícolas.

Com rigor metodológico, o projeto separa o **artefato analítico construído e medido** da **especificação conceitual futura do produto móvel**:

| Dimensão | O que foi Construído e Validado neste Repositório (`Etapa 4/`) | O que é Projeto Arquitetural Proposto (Roadmap TRL 4 → TRL 6) |
| :--- | :--- | :--- |
| **Software e Código** | Esteira analítica completa em **4 Jupyter Notebooks (`.ipynb`)** em [`Etapa 4/notebooks/`](Etapa%204/notebooks/), com saídas salvas e *fallback* para execução imediata pós-`git clone`. | Aplicativo **AgroSense Mobile** (especificação conceitual em Flutter + ONNX Runtime/TFLite + SQLite + GPS assíncrono). Nenhum APK/IPA compilado nesta etapa. |
| **Modelo Neural** | Rede **MobileNetV2** pré-treinada na ImageNet e ajustada em PyTorch para **38 classes**, com perda ponderada inversa e checkpoint persistido (`26,38 MB`). | Exportação quantizada (INT8/FP16) para execução local *off-line* em *smartphones*. |
| **Validação e Métricas** | **99,41% de acurácia Top-1** no teste cego *in-domain* ($N = 8.146$, PlantVillage), auditoria das 48 confusões e validação do limiar Softmax a 85% (**99,79%** de acurácia aceita). | Coleta piloto de fotos reais em lavoura (*in-the-wild*), incluindo doenças de soja, remoção/diversificação de fundo e validação agronômica em campo. |
| **Latência** | Medida empiricamente em **hardware desktop local**: **1,91 ms/img** em lote GPU NVIDIA RTX 3060 (**523,4 img/s**); inferência unitária (`batch_size=1`) de **3,75 ms** na GPU e **7,47 ms** na CPU `x86_64`. | **Não medida em smartphone físico.** A meta de **< 50 ms on-device** é requisito de projeto para a fase futura de prototipagem mobile. |

---

## 3. Guia Passo a Passo de Replicação (`git clone`)

### 3.1. Clonagem e Instalação do Ambiente (Python 3.12+)

Em qualquer computador (Windows, Linux ou macOS), execute no terminal:

```bash
# 1. Clone o repositório e acesse a pasta raiz
git clone https://github.com/henri-sarmento/ProjetoAplicado2.git
cd ProjetoAplicado2

# 2. Crie e ative um ambiente virtual limpo
python -m venv .venv
# No Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# No Windows (CMD):
.\.venv\Scripts\activate.bat
# No Linux / macOS:
source .venv/bin/activate

# 3. Instale todas as dependências
pip install -r requirements.txt
```

> **Dica (GPU NVIDIA Opcional):** O projeto roda perfeitamente em **CPU** ou **GPU**. Caso possua GPU NVIDIA com CUDA 12.4 e deseje aceleração por hardware:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
> ```

### 3.2. Como Executar a Replicação Final na `Etapa 4/`

Para não sobrecarregar o Git com mais de 800 MB de imagens brutas, o diretório de fotos originais (`plantvillage dataset/color/`) é ignorado no `.gitignore`, enquanto o catálogo completo auditado (`Etapa 4/data/metadados_plantvillage.csv`), o modelo treinado (`Etapa 4/models/mobilenetv2_plantvillage.pth`) e as métricas do teste cego (`Etapa 4/reports/metrics/`) **já estão versionados no repositório**.

Por isso, você pode reproduzir o projeto de duas formas:

#### Modo A — Replicação Imediata Pós-Clone (100% Off-line, Sem Baixar 800 MB de Imagens)
Basta abrir os notebooks da pasta **`Etapa 4/notebooks/`** no JupyterLab, VS Code ou Google Colab:

```bash
jupyter lab
```

Todos os cadernos de `01` a `04` possuem detecção automática de ambiente (**Modo Catálogo**) e executam do início ao fim sem erro mesmo sem as fotos brutas:

1. **[`Etapa 4/notebooks/01_eda_e_metadados.ipynb`](Etapa%204/notebooks/01_eda_e_metadados.ipynb):** Carrega `Etapa 4/data/metadados_plantvillage.csv` ($N = 54.305$), audita a integridade das 38 classes e 14 culturas e regenera em 300 DPI os gráficos de distribuição de classes (`eda_distribuicao_classes.png`), status sanitário (`eda_status_sanitario.png`) e assinatura RGB (`eda_distribuicao_rgb.png`).
2. **[`Etapa 4/notebooks/02_preparacao_e_treinamento_mobilenetv2.ipynb`](Etapa%204/notebooks/02_preparacao_e_treinamento_mobilenetv2.ipynb):** Executa a divisão estratificada determinística 70/15/15 (`seed=42`, 0% *data leakage*), calcula os pesos inversos da `CrossEntropyLoss`, carrega o checkpoint `Etapa 4/models/mobilenetv2_plantvillage.pth` (`USE_EXISTING_CHECKPOINT = True`) e gera `curvas_aprendizado.png`.
3. **[`Etapa 4/notebooks/03_avaliacao_metricas_e_diagnostico.ipynb`](Etapa%204/notebooks/03_avaliacao_metricas_e_diagnostico.ipynb):** Carrega as métricas do teste cego ($N = 8.146$), **mede ao vivo a latência unitária (`batch_size=1`) na CPU/GPU do seu computador** e executa a memória de cálculo financeira do ROI (1.000 ha).
4. **[`Etapa 4/notebooks/04_geracao_figuras_relatorio.ipynb`](Etapa%204/notebooks/04_geracao_figuras_relatorio.ipynb):** Gera programaticamente em 300 DPI os diagramas de Arquitetura Edge AI (`figura_arquitetura_edge_ai.png`), Business Model Canvas (`figura_canvas_negocios.png`) e Arco Narrativo em 3 Atos (`figura_storytelling_arco.png`).


Se preferir reexecutar via terminal (`nbconvert`):

```bash
jupyter nbconvert --to notebook --execute --inplace "Etapa 4/notebooks/01_eda_e_metadados.ipynb"
jupyter nbconvert --to notebook --execute --inplace "Etapa 4/notebooks/02_preparacao_e_treinamento_mobilenetv2.ipynb"
jupyter nbconvert --to notebook --execute --inplace "Etapa 4/notebooks/03_avaliacao_metricas_e_diagnostico.ipynb"
jupyter nbconvert --to notebook --execute --inplace "Etapa 4/notebooks/04_geracao_figuras_relatorio.ipynb"
```

#### Modo B — Reexecução Completa com as 54.305 Imagens Brutas do PlantVillage (Opcional)
Caso deseje reexecutar a varredura física dos JPEGs (Notebook 01), retreinar as 5 épocas do zero (`USE_EXISTING_CHECKPOINT = False` no Notebook 02) ou rodar a inferência em lote nas 8.146 imagens do teste cego (Notebook 03):
1. Baixe o subconjunto colorido (`color`) do dataset público **PlantVillage** (54.305 imagens divididas nas 38 pastas originais);
2. Posicione a pasta em `ProjetoAplicado2/plantvillage dataset/color/` (os notebooks detectam automaticamente esse caminho e ativam o modo de leitura completa de imagens).

---

## 4. Estrutura Completa de Pastas do Repositório

```text
ProjetoAplicado2/
├── README.md                                          # Documentação central e guia de replicação
├── requirements.txt                                   # Dependências unificadas do projeto (Python 3.12+)
├── .gitignore                                         # Regras de exclusão de ambientes e imagens brutas (>800 MB)
├── pytest.ini                                         # Configuração da suíte histórica de testes
│
├── docs/                                              # ENTREGAS OFICIAIS CONSOLIDADAS (ETAPAS 1 A 4)
│   ├── Relatorio_Etapa4_AgroSense_ABNT.pdf            # ★ Relatório Técnico Final da Etapa 4 (PDF)
│   ├── Relatorio_Etapa4_AgroSense_ABNT.docx           # ★ Relatório Técnico Final da Etapa 4 (Word ABNT)
│   ├── Apresentacao_Etapa4_AgroSense.pptx             # ★ Apresentação Executiva Final (10 Slides 16:9)
│   ├── Relatorio_Etapa3_AgroSense_ABNT.docx / .pdf    # Histórico: Relatório da Etapa 3
│   ├── A2 - Aplicando conhecimento.docx / .pdf        # Histórico: Relatório da Etapa 2
│   └── A1 - Aplicando conhecimento.docx / .pdf        # Histórico: Relatório da Etapa 1
│
├── Etapa 4/                                           # ★ DIRETÓRIO CANÔNICO PARA REPLICAÇÃO FINAL
│   ├── notebooks/                                     # 4 Jupyter Notebooks (.ipynb) executados e autocontidos
│   │   ├── 01_eda_e_metadados.ipynb                   # Catálogo de 54.305 imagens e gráficos exploratórios (EDA)
│   │   ├── 02_preparacao_e_treinamento_mobilenetv2.ipynb # Split 70/15/15, perda ponderada e MobileNetV2
│   │   ├── 03_avaliacao_metricas_e_diagnostico.ipynb  # Teste cego N=8.146, 48 confusões, limiar 85%, latência e ROI
│   │   └── 04_geracao_figuras_relatorio.ipynb         # Diagramas de Arquitetura, Canvas e Storytelling (300 DPI)
│   ├── data/
│   │   ├── metadados_plantvillage.csv                 # Catálogo auditado (54.305 registros, 38 classes)
│   │   └── metadados_plantvillage.csv.gz              # Versão compactada
│   ├── models/
│   │   └── mobilenetv2_plantvillage.pth               # Checkpoint MobileNetV2 treinado (26,38 MB)
│   └── reports/
│       ├── metrics/
│       │   ├── metricas_resumo.json                   # Métricas do teste cego e benchmark de latência desktop
│       │   └── classification_report.csv              # Precisão, Revocação e F1 das 38 classes
│       └── figures/                                   # 12 figuras em 300 DPI (EDA, treino, teste, Canvas e Arco)
│
├── Etapa 3/                                           # Registro histórico da Etapa 3
├── Etapa 2/                                           # Registro histórico da Etapa 2
└── Etapa 1/                                           # Registro histórico da Etapa 1
```