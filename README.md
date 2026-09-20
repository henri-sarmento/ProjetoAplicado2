# AgroSense Analytics — Inteligência Fitossanitária Móvel (Edge AI)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-ee4c2c.svg)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.21.0-orange.svg)](https://pytorch.org/vision/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Etapa 2 Concluída](https://img.shields.io/badge/Status-Etapa%202%20Conclu%C3%ADda-brightgreen.svg)]()

> **Projeto Aplicado II — Ciência de Dados**  
> **Universidade Presbiteriana Mackenzie** — Faculdade de Computação e Informática (FCI)  
> **Equipe:** Henrique Sarmento, Felipe Magalhães de Melo Costa, Jefferson Candido Moreira, Karla Maria Ramos da Silva  

---

## 1. Proposta de Valor
A **AgroSense Analytics** é uma agritech voltada à democratização da agricultura de precisão. O projeto desenvolve uma solução de **Edge AI (Inteligência Artificial embarcada)** para dispositivos móveis (Android), capaz de diagnosticar instantaneamente 38 classes de fitopatologias a partir de fotografias foliares tiradas diretamente pelo agricultor em campo, funcionando **100% offline** e sem necessidade de conexão com a internet.

---

## 2. Métricas de Eficácia no Conjunto de Teste Inédito ($N = 8.146$)
O modelo supervisionado foi avaliado no conjunto de teste independente (15% das 54.305 imagens), com garantia matemática de ausência de vazamento de dados (*Zero Data Leakage*):

| Métrica de Avaliação | Resultado Obtido | Significado Prático para a AgroSense |
| :--- | :---: | :--- |
| **Acurácia Global (Top-1)** | **99,41%** | 8.098 acertos exatos em 8.146 imagens de teste cego. |
| **Acurácia Top-3** | **99,96%** | O diagnóstico real figura entre as 3 principais sugestões do app. |
| **Acurácia Top-5** | **99,98%** | Risco quase nulo de omissão completa da patologia. |
| **Macro F1-Score** | **99,19%** | Eficácia homogênea entre classes majoritárias e minoritárias. |
| **Weighted F1-Score** | **99,41%** | Desempenho ponderado pela representatividade de cada cultura. |
| **Revocação da Classe Minoritária** | **100,00%** | **23 de 23 amostras de `Potato___healthy`** detectadas com 100% de precisão. |
| **Tempo de Treinamento** | **19,24 min** | 5 épocas em GPU NVIDIA RTX 3060 (12 GB VRAM) com PyTorch CUDA 12.4. |

---

## 3. Arquitetura Técnica da Solução
* **Topologia:** **MobileNetV2** (Sandler et al., 2018) via *Transfer Learning* com pesos pré-treinados no ImageNet-1k.
* **Mecanismo de Eficiência:** **Convoluções Separáveis em Profundidade (*Depthwise Separable Convolutions*)** e Blocos Residuais Invertidos com Gargalo Linear, reduzindo o custo computacional de operações (FLOPs) em aproximadamente **9 vezes** ($\text{Razão} \approx 1/9$) em relação a arquiteturas clássicas como ResNet-50.
* **Footprint:** Apenas **~3,5 milhões de parâmetros** e checkpoint de **26,4 MB**, viabilizando inferência rápida (< 50 ms) em smartphones de entrada.
* **Otimização:** Otimizador **AdamW** ($\text{lr} = 10^{-4}$, $\text{weight\_decay} = 10^{-4}$), agendamento adaptativo com `ReduceLROnPlateau` e **CrossEntropyLoss ponderada** com pesos inversamente proporcionais à frequência das classes para neutralizar o desbalanceamento de 36:1 do dataset.

---

## 4. Estrutura do Repositório
```text
ProjetoAplicado2/
├── data/
│   ├── metadados_plantvillage.csv      # Catálogo estruturado com 54.305 linhas e 19 colunas
│   └── metadados_plantvillage.csv.gz   # Versão compactada para economia de tráfego
├── docs/
│   ├── A1 - Aplicando conhecimento.pdf # Documento oficial entregue na Etapa 1
│   ├── Modelo de entrega.docx          # Template base acadêmico da instituição
│   ├── Relatorio_Etapa_2_Produto_Analitico.md # Relatório técnico completo em Markdown
│   └── Relatorio_Etapa2_AgroSense.docx # Relatório acadêmico compilado da Etapa 2 (ABNT)
├── models/
│   └── mobilenetv2_plantvillage.pth    # Checkpoint ótimo da MobileNetV2 treinada (26,4 MB)
├── notebooks/
│   ├── Etapa_2_Analise_Exploratoria_e_Dicionario.ipynb # Notebook 1: EDA e Dicionário de Dados
│   └── Etapa_2_Treinamento_Modelo_e_Metricas.ipynb     # Notebook 2: Treino e Avaliação PyTorch
├── reports/
│   ├── figures/                        # 7 figuras técnicas exportadas em 300 DPI
│   │   ├── eda_distribuicao_classes.png
│   │   ├── eda_status_sanitario.png
│   │   ├── eda_mosaico_patologias.png
│   │   ├── eda_distribuicao_rgb.png
│   │   ├── curvas_aprendizado.png
│   │   ├── matriz_confusao.png
│   │   └── galeria_inferencia.png
│   └── metrics/
│       ├── classification_report.csv   # Relatório de precisão, recall e F1 por patologia
│       └── metricas_resumo.json        # Resumo executivo estruturado em JSON
├── src/
│   ├── extract_metadata.py             # Extração concorrente de metadados das imagens
│   ├── train_etapa2.py                 # Script de treinamento supervisionado MobileNetV2
│   ├── evaluate_metrics.py             # Avaliação independente no teste e métricas completas
│   ├── export_eda_figures.py           # Renderização e exportação das 4 figuras de EDA
│   └── convert_md_to_docx.py           # Compilação do relatório ABNT para Microsoft Word
├── tests/
│   ├── conftest.py                     # Fixtures e configuração do Pytest
│   ├── run_tests.py                    # Runner automatizado da suíte de testes
│   ├── test_etapa2_pipeline.py         # Testes funcionais e asserções de zero leakage
│   └── test_empirical_challenger.py    # Testes adversariais e casos de borda
├── .gitignore                          # Regras de exclusão de artefatos e dados brutos
├── pytest.ini                          # Configuração de execução do Pytest
├── README.md                           # Documentação central do projeto
└── requirements.txt                    # Dependências de produção do projeto
```

---

## 5. Instruções Rápidas de Reprodução

### 5.1 Instalação do Ambiente
Clone o repositório e instale as dependências:
```bash
git clone https://github.com/henri-sarmento/ProjetoAplicado2.git
cd ProjetoAplicado2
pip install -r requirements.txt
```

> **Dica para aceleração CUDA:** Caso possua GPU NVIDIA, instale a versão com suporte a CUDA 12.4:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
> ```

### 5.2 Avaliação das Métricas do Modelo (Sem Re-treinamento)
Execute a inferência cega sobre o conjunto de teste utilizando o checkpoint persistido:
```bash
python src/evaluate_metrics.py
```
*Gera os relatórios `reports/metrics/classification_report.csv` e `reports/metrics/metricas_resumo.json`.*

### 5.3 Exportação das Figuras em Alta Resolução (300 DPI)
Para regerar os gráficos de EDA:
```bash
python src/export_eda_figures.py
```

### 5.4 Compilação do Relatório Word Oficial
Para compilar o documento ABNT formatado:
```bash
python src/convert_md_to_docx.py
```

### 5.5 Execução dos Testes Automatizados
```bash
pytest tests/
```

---

## 6. Autores e Identificação Institucional
* **Instituição:** Universidade Presbiteriana Mackenzie
* **Curso:** Ciência de Dados — 4º Semestre
* **Disciplina:** Projeto Aplicado II
* **Equipe de Desenvolvimento:**
  * Henrique Sarmento
  * Felipe Magalhães de Melo Costa
  * Jefferson Candido Moreira
  * Karla Maria Ramos da Silva
