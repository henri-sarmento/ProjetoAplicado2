# AgroSense Analytics — Inteligência Fitossanitária Móvel (Edge AI)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-ee4c2c.svg)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.21.0-orange.svg)](https://pytorch.org/vision/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Etapa 3 Concluída](https://img.shields.io/badge/Status-Etapa%203%20Conclu%C3%ADda-brightgreen.svg)]()

> **Projeto Aplicado II — Ciência de Dados**  
> **Universidade Presbiteriana Mackenzie** — Faculdade de Computação e Informática (FCI)  
> **Integrantes da Equipe:**  
> • Henrique Sarmento  
> • Felipe Magalhães de Melo Costa  
> • Jefferson Candido Moreira  
> • Karla Maria Ramos da Silva  

---

## 1. Visão Geral e Proposta de Valor
A **AgroSense Analytics** é uma agritech concebida para democratizar o acesso à agricultura de precisão e mitigar os gargalos fitossanitários do campo brasileiro. Desenvolvemos uma solução de **Edge AI (Inteligência Artificial embarcada)** para dispositivos móveis, capaz de diagnosticar instantaneamente 38 classes de patologias em 14 culturas agrícolas vitais diretamente a partir de fotografias foliares capturadas no talhão.

O produto opera **100% off-line**, dispensando qualquer conectividade celular (4G/5G) ou acesso a servidores em nuvem no momento da inferência, superando o principal obstáculo tecnológico do agronegócio nacional: mais de 70% das lavouras não possuem sinal de internet móvel.

---

## 2. Diagnóstico Formal e Métricas no Teste Cego ($N = 8.146$)
A eficácia do modelo foi rigorosamente validada no conjunto de teste independente (15% das 54.305 imagens do acervo PlantVillage), com partição estratificada e garantia matemática de **ausência total de vazamento de dados (*Zero Data Leakage*)**:

| Métrica de Avaliação | Resultado Obtido | Significado Prático para a AgroSense |
| :--- | :---: | :--- |
| **Acurácia Global (Top-1)** | **99,41%** | 8.098 acertos exatos em 8.146 imagens inéditas de teste. |
| **Acurácia Top-3** | **99,96%** | O diagnóstico real figura quase invariavelmente entre as 3 principais alternativas. |
| **Acurácia Top-5** | **99,98%** | Risco praticamente nulo de omissão da patologia no aplicativo. |
| **Macro F1-Score** | **99,19%** | Eficácia uniforme entre classes abundantes e patologias raras. |
| **Weighted F1-Score** | **99,41%** | Desempenho ponderado pela representatividade de cada cultura. |
| **Sensibilidade na Minoria** | **100,00%** | **23 de 23 amostras de `Potato___healthy`** detectadas com 100% de precisão. |
| **Tempo de Inferência Local** | **< 50 ms** | Resposta instantânea on-device via NPU/CPU de smartphones convencionais. |
| **Footprint de Armazenamento** | **26,38 MB** | Grafo neural compilado viabilizando download ágil e baixo consumo de memória. |

---

## 3. Especificação do Produto e Modelo de Negócios (B2B SaaS)

### 3.1 Arquitetura da Solução Edge AI
* **Motor Neural On-Device:** Rede convolucional **MobileNetV2** adaptada com convoluções separáveis em profundidade (*Depthwise Separable Convolutions*) e blocos residuais invertidos, reduzindo o custo computacional (FLOPs) em ~9x frente a redes clássicas (ResNet-50).
* **Blindagem Fitossanitária:** Limiar de confiança ativo em 85%. Inferências abaixo dessa probabilidade acionam recomendação protocolar para envio de amostra a laboratório fitopatológico credenciado.
* **Sincronização Assíncrona Opcional:** Coleta georreferenciada de laudos via GPS local. Assim que o técnico retorna à sede com conectividade, os registros alimentam o dashboard epidemiológico regional da cooperativa.

### 3.2 Impacto Financeiro e Projeção de ROI
* **Economia Direta de Defensivos:** Redução de **R$ 180 a R$ 350 por hectare** ao substituir pulverizações calendarizadas por intervenções cirúrgicas de precisão.
* **Resgate de Produtividade:** Prevenção de quebra de safra estimada entre **5% e 15%** mediante diagnóstico e contenção nas primeiras 48 horas de infestação.
* **Aderência ESG:** Redução substancial de resíduos químicos no solo e nos lençóis freáticos, alinhando a operação aos ODS 2 (Fome Zero e Agricultura Sustentável) e ODS 12 (Consumo e Produção Responsáveis) da ONU.

---

## 4. Estrutura Modular do Repositório

O repositório está organizado de forma cumulativa e auditável por etapas de desenvolvimento:

```text
ProjetoAplicado2/
├── docs/                                  # Entregas oficiais consolidadas do projeto
│   ├── A1 - Aplicando conhecimento.docx / .pdf  # Etapa 1: Formulação do Problema
│   ├── A2 - Aplicando conhecimento.docx / .pdf  # Etapa 2: Modelagem Preditiva
│   └── Relatorio_Etapa3_AgroSense_ABNT.docx / .pdf # Etapa 3: Produto Analítico e Negócios
├── Etapa 1/                               # Concepção, justificativa e referencial teórico
│   ├── A1 - Aplicando conhecimento.docx
│   └── A1 - Aplicando conhecimento.pdf
├── Etapa 2/                               # Pipeline de Machine Learning e Validação Cega
│   ├── data/                              # Metadados das 54.305 imagens auditadas
│   ├── models/                            # Checkpoint treinado da MobileNetV2 (26,38 MB)
│   ├── notebooks/                         # Notebooks exploratórios e de treinamento
│   ├── reports/                           # Figuras de EDA (300 DPI) e relatórios de métricas
│   ├── src/                               # Scripts modulares (treinamento, EDA e métricas)
│   ├── tests/                             # Suíte automatizada de testes e asserções antifalha
│   └── Relatorio_Etapa2_AgroSense_ABNT.docx
├── Etapa 3/                               # Produto Analítico, Viabilidade e Storytelling
│   ├── data/                              # Catálogo de metadados sincronizado
│   ├── docs/                              # Plano de trabalho detalhado da Etapa 3
│   ├── models/                            # Grafo MobileNetV2 validado
│   ├── notebooks/                         # Esteira analítica interativa completa
│   │   ├── 01_eda_e_metadados.ipynb
│   │   ├── 02_preparacao_e_treinamento_mobilenetv2.ipynb
│   │   ├── 03_avaliacao_metricas_e_diagnostico.ipynb
│   │   └── 04_geracao_figuras_relatorio.ipynb # Customização visual de BMC e Storytelling
│   ├── reports/                           # Artefatos visuais e métricas consolidadas
│   │   ├── figures/                       # BMC, Arco Narrativo, Arquitetura e EDA
│   │   └── metrics/                       # classification_report.csv e metricas_resumo.json
│   ├── relatorio_etapa3.md                # Fonte textual do relatório técnico
│   └── Relatorio_Etapa3_AgroSense_ABNT.docx # Relatório final formal segundo normas ABNT
├── .gitignore                             # Regras de exclusão de binários e dados volumosos
├── pytest.ini                             # Configuração da suíte Pytest
├── README.md                              # Documentação central do repositório
└── requirements.txt                       # Dependências de produção, pesquisa e notebooks
```

---

## 5. Instruções Rápidas de Reprodução

### 5.1 Configuração do Ambiente Virtual
Clone o repositório e instale as dependências:
```bash
git clone https://github.com/henri-sarmento/ProjetoAplicado2.git
cd ProjetoAplicado2
python -m venv .venv
# Ativação no Windows:
.venv\Scripts\activate
# Ativação no Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

> **Aceleração via GPU (NVIDIA CUDA 12.4):** Caso utilize ambiente com placa NVIDIA, instale o build correspondente do PyTorch:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
> ```

### 5.2 Execução dos Testes Automatizados da Etapa 2
Para executar as asserções de qualidade, integridade dos tensores e ausência de *data leakage*:
```bash
pytest "Etapa 2/tests"
```

### 5.3 Auditoria Independente de Métricas (Sem Necessidade de Retreino)
Execute a inferência cega sobre o conjunto de teste utilizando o checkpoint persistido:
```bash
python "Etapa 2/src/evaluate_metrics.py"
```

### 5.4 Execução da Esteira da Etapa 3 via Jupyter
Inicie o servidor Jupyter para explorar ou reexecutar os notebooks da esteira analítica:
```bash
jupyter lab
```
Navegue até a pasta `Etapa 3/notebooks/` para acessar:
* `01_eda_e_metadados.ipynb`: Análise exploratória e extração de metadados foliares.
* `02_preparacao_e_treinamento_mobilenetv2.ipynb`: Pipeline PyTorch e treinamento supervisionado com função de perda ponderada.
* `03_avaliacao_metricas_e_diagnostico.ipynb`: Matriz de confusão e métricas no teste cego.
* `04_geracao_figuras_relatorio.ipynb`: Geração e customização visual interativa do Business Model Canvas, Arco Narrativo e Arquitetura Edge AI.

---

## 6. Arco Narrativo de Data Storytelling (Pitch Executivo)

* **Ato 1: O Prejuízo Invisível no Talhão (O Problema)**  
  O agronegócio brasileiro perde anualmente mais de R$ 55 bilhões devido a pragas e patologias foliares. A inspeção visual humana é escassa e sujeita a diagnósticos tardios. Quando o agrônomo tenta recorrer a ferramentas digitais, depara-se com o fato de que mais de 70% das lavouras não possuem sinal 4G/5G, exibindo a tela de *"Sem conexão"*.

* **Ato 2: AgroSense Mobile — Inteligência Artificial On-Device Sem Internet (A Proposta de Valor)**  
  O AgroSense coloca um laboratório fitopatológico no bolso do agrônomo. O diagnóstico ocorre em menos de 50 milissegundos através da câmera, operando 100% off-line. Essa agilidade é sustentada por uma rede neural convolucional leve (MobileNetV2, 26,38 MB), calibrada para 38 classes patológicas e comprovada com 99,41% de acurácia em teste cego.

* **Ato 3: Da Detecção Precoce ao Bolso — Economia e Inteligência Regional (O Resultado)**  
  A intervenção em até 48 horas resgata de 5% a 15% da produtividade que seria perdida e gera economia direta de R$ 180 a R$ 350 por hectare em defensivos. Para as cooperativas, a telemetria assíncrona georreferenciada viabiliza mapas de calor regionais e prevenção epidemiológica, com pleno alinhamento ESG.

---

## 7. Licença e Referências Técnicas
* **Licença:** Distribuído sob a licença [MIT](LICENSE).
* **Base de Dados:** Hughes, D., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics*. arXiv:1511.08060 (PlantVillage).
* **Arquitetura Base:** Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR 2018.
