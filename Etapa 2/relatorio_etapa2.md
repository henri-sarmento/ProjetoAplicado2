# PROJETO APLICADO II — CIÊNCIA DE DADOS
## ETAPA 2: DEFINIÇÃO DO PRODUTO ANALÍTICO E MODELAGEM PREVENTIVA

**Organização:** AgroSense Analytics[cite: 4]  
**Repositório Oficial no GitHub:** [https://github.com/henri-sarmento/ProjetoAplicado2](https://github.com/henri-sarmento/ProjetoAplicado2)[cite: 5]  
**Instituição:** Universidade Presbiteriana Mackenzie[cite: 9]  

---

### 1. INTRODUÇÃO, REPOSITÓRIO E STACK TECNOLÓGICA

#### 1.1 Contexto e Proposta de Valor
A **AgroSense Analytics** é uma *agtech* focada em democratizar a agricultura de precisão através do fornecimento de inteligência computacional para produtores rurais[cite: 4]. O escopo central deste projeto consiste no desenvolvimento de um sistema ponta a ponta de visão computacional capaz de identificar, com precisão e em tempo real, 38 condições fitossanitárias em folhas agrícolas diretamente no campo[cite: 4, 5].

#### 1.2 Repositório Git e Rastreabilidade
O desenvolvimento colaborativo do projeto é centralizado no GitHub[cite: 6, 8], garantindo o versionamento contínuo de código, artefatos e relatórios no endereço:
`https://github.com/henri-sarmento/ProjetoAplicado2`[cite: 5].

#### 1.3 Ecossistema de Bibliotecas Python
Para estruturar o pipeline analítico, foram padronizados os seguintes pacotes em ambiente Python 3.10+[cite: 6, 8]:
* **PyTorch e Torchvision:** motores centrais para modelagem profunda, carregamento de tensores, implementação do *Transfer Learning* e inferência da arquitetura MobileNetV2;
* **Pillow (PIL) e OpenCV:** decodificação, manipulação matricial e tratamento defensivo do padrão cromático das imagens[cite: 7];
* **Scikit-Learn:** suporte para particionamento estratificado sem vazamento de dados (*Zero Data Leakage*) e cômputo das matrizes de contingência multiclasse;
* **Pandas e NumPy:** estruturação do catálogo de metadados, operações lineares vetorizadas e agregação de dados tabulares[cite: 7];
* **Matplotlib e Seaborn:** geração de evidências visuais, curvas de convergência e mapas de calor analíticos.

---

### 2. ANÁLISE EXPLORATÓRIA E BASE DE DADOS

#### 2.1 Visão Geral do PlantVillage Dataset
O acervo de dados utilizado é o *PlantVillage Dataset* (hospedado no Kaggle), compreendendo 54.305 imagens digitais de folhas distribuídas em 38 classes patológicas e botânicas[cite: 4, 5]. O acervo apresenta resolução nativa de $256 \times 256$ pixels no espaço tridimensional RGB[cite: 4, 7].

#### 2.2 Dicionário de Dados do Acervo Estruturado
Para viabilizar a exploração computacional no Python, os metadados de leitura das pastas foram consolidados no arquivo `metadados_plantvillage.csv`[cite: 7]:

| Nome da Coluna | Descrição Técnica do Atributo | Tipo de Dado |
| :--- | :--- | :---: |
| `caminho_arquivo` | Caminho relativo do diretório e nome do arquivo no disco[cite: 7]. | String |
| `matriz_pixels` | Tensor de intensidade nos canais de cor Red, Green e Blue ($3 \times 224 \times 224$)[cite: 4, 7]. | Tensor / Float |
| `rotulo_classe` | Rótulo categórico formal combinando espécie vegetal e patologia (38 classes)[cite: 4, 7]. | Categórico / Int |
| `especie_planta` | Espécie agrícola isolada por decomposição do rótulo de origem[cite: 7]. | String |
| `condicao_saude` | Diagnóstico fitossanitário isolado (sadia versus patologia específica)[cite: 7]. | String |

#### 2.3 Diagnóstico de Desbalanceamento Amostral
A exploração revelou uma assimetria severa na distribuição de frequências entre as 38 classes:
* **Classe Majoritária:** *Orange Citrus Greening* com 5.507 amostras;
* **Classe Minoritária:** *Potato healthy* com apenas 152 amostras.

Essa discrepância estabelece uma razão demográfica de **36,2 : 1**, impondo a necessidade de tratamento matemático contra o viés de predição nas fases de treinamento e avaliação.

#### 2.4 Inviabilidade de Modelagem Preditiva Tabular
A tabela `metadados_plantvillage.csv` não pôde ser utilizada como entrada direta para modelos tabulares clássicos (como Random Forest ou XGBoost) devido a dois impedimentos técnicos:
1. **Vazamento Direto do Rótulo (*Target Leakage*):** variáveis derivadas como `especie_planta` e `condicao_saude` foram extraídas da decomposição léxica do próprio caminho das pastas[cite: 7]. Treinar modelos sobre essas colunas geraria uma acurácia espúria de 100%, sem que o algoritmo de fato analisasse a folha[cite: 7]. Em campo, a câmera do produtor rural não recebe metadados — a imagem é a única entrada real[cite: 4, 7].
2. **Perda da Estrutura Espacial dos Sintomas:** os sinais fitopatológicos (halos concêntricos, pontuações necróticas e manchas angulares) residem exclusivamente na topologia espacial da matriz $\mathbf{X} \in \mathbb{R}^{3 \times 224 \times 224}$[cite: 4, 7]. Métricas tabulares agregadas (como média de canal de cor ou tamanho de arquivo) destroem as relações de vizinhança entre os pixels, cegando o modelo quanto à morfologia da doença e tornando mandatório o uso de Redes Neurais Convolucionais (CNNs)[cite: 7].

---

### 3. TRATAMENTO DA BASE DE DADOS E PIPELINE DE ENGENHARIA

#### 3.1 Particionamento Estratificado (70% Treino, 15% Validação e 15% Teste)
O particionamento do acervo ($N = 54.305$) foi conduzido sob amostragem estratificada para preservar com exatidão a distribuição de probabilidade marginal $P(Y = c)$ das 38 classes[cite: 4, 7]:

| Subconjunto de Dados | Proporção | Volume Absoluto | Função no Pipeline da AgroSense |
| :--- | :---: | :---: | :--- |
| **Treinamento (`df_train`)** | 70,0% | 38.013 amostras | Otimização dos pesos sinápticos via retropropagação de gradientes com AdamW. |
| **Validação (`df_val`)** | 15,0% | 8.146 amostras | Ajuste de hiperparâmetros, acompanhamento do *scheduler* e seleção do melhor *checkpoint*. |
| **Teste Inédito (`df_test`)** | 15,0% | 8.146 amostras | Avaliação cega final para mensurar a capacidade de generalização do modelo. |
| **Total Consolidado** | **100,0%** | **54.305 amostras** | Cobertura integral das 38 classes em todas as partições[cite: 4, 7]. |

A integridade do isolamento foi validada por asserções de conjuntos: a interseção entre os índices das três partições é rigorosamente vazia ($\mathcal{I}_{\text{treino}} \cap \mathcal{I}_{\text{val}} \cap \mathcal{I}_{\text{teste}} = \emptyset$), garantindo a ausência total de vazamento de dados (*Zero Data Leakage*).

#### 3.2 Pipeline de Pré-Processamento e Aumento de Dados (*Data Augmentation*)
O fluxo de tratamento de imagens implementou as seguintes regras de engenharia:
* **Padronização Cromática Defensiva (Tratamento de Anomalia RGBA):** a análise exploratória detectou que 54.304 imagens estavam no formato RGB padrão e apenas 1 arquivo continha canal alfa de transparência (formato RGBA de 4 canais). Para evitar a quebra dimensional do tensor de entrada na primeira camada convolucional (`RuntimeError`), padronizou-se o carregamento via biblioteca Pillow forçando a conversão direta (`convert('RGB')`), descartando a transparência sem perda do tecido foliar.
* **Normalização com Parâmetros do ImageNet:** padronização afim $x' = \frac{x - \mu}{\sigma}$ utilizando as médias $\boldsymbol{\mu} = [0,485;\, 0,456;\, 0,406]$ e desvios $\boldsymbol{\sigma} = [0,229;\, 0,224;\, 0,225]$ pré-computados do ImageNet, preservando o alinhamento com os filtros convolucionais pré-treinados.
* **Aumento de Dados (*Data Augmentation*) Confinado ao Treino:** aplicação estocástica em tempo de execução de redimensionamento para $224 \times 224$, espelhamentos horizontais ($p=0,5$) e verticais ($p=0,2$), rotações contínuas de até $\pm 20^\circ$ e variações de iluminação via *ColorJitter* (15% em brilho, contraste e saturação; 5% em matiz)[cite: 4].
* **Determinismo em Validação e Teste:** nos conjuntos de validação e teste, aplicou-se apenas o redimensionamento bilinear e a normalização de tensores, garantindo que as medições reflitam a morfologia autêntica das folhas[cite: 4, 7].

---

### 4. FUNDAMENTAÇÃO TEÓRICA DA MODELAGEM

#### 4.1 Aprendizado por Transferência (*Transfer Learning*)
Para viabilizar a convergência rápida sem a necessidade de milhões de imagens adicionais, utilizou-se o paradigma de *Transfer Learning*. A rede aproveita os extratores visuais genéricos aprendidos sobre a base ImageNet-1k, substituindo o classificador original de 1.000 classes por uma camada linear customizada (`Linear(in_features=1280, out_features=38)`) ajustada ponta a ponta com taxa de aprendizado controlada ($\eta = 10^{-4}$).

#### 4.2 Seleção da Arquitetura MobileNetV2 para *Edge AI*
A escolha da **MobileNetV2** (Sandler et al., 2018) reflete a necessidade operacional do produto: execução instantânea local em smartphones intermediários de produtores no campo (*Edge AI*), sem depender de conectividade com a internet[cite: 4]. 

Enquanto redes densas (como a ResNet-50 com ~25,6M parâmetros) exigem alto consumo de bateria e memória RAM, a MobileNetV2 entrega acurácia comparável operando com apenas **~3,5 milhões de parâmetros** e latência inferior a **50 ms** por quadro.

#### 4.3 Convoluções Separáveis em Profundidade (*Depthwise Separable Convolutions*)
Introduzida na família MobileNet por **Howard et al. (2017)**, a convolução separável em profundidade divide a filtragem espacial da combinação linear de canais:
* **Convolução Clássica:** $\text{Custo}_{\text{conv}} = D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F$
* **Etapa *Depthwise* (filtragem espacial por canal):** $\text{Custo}_{\text{depthwise}} = D_K \cdot D_K \cdot M \cdot D_F \cdot D_F$
* **Etapa *Pointwise* (projeção linear $1 \times 1$):** $\text{Custo}_{\text{pointwise}} = 1 \cdot 1 \cdot M \cdot N \cdot D_F \cdot D_F$

Somando ambas as operações, a razão de custo computacional frente à camada clássica é dada por:
$$\text{Razão} = \frac{D_K^2 \cdot M \cdot D_F^2 + M \cdot N \cdot D_F^2}{D_K^2 \cdot M \cdot N \cdot D_F^2} = \frac{1}{N} + \frac{1}{D_K^2}$$

Para filtros espaciais $3 \times 3$ ($D_K = 3$) e grande volume de canais de saída ($N \gg 9$):
$$\text{Razão} \approx \frac{1}{3^2} = \frac{1}{9} \approx 0,11$$

Essa formulação comprova que a arquitetura consome entre **8 e 9 vezes menos operações multiplicativas (FLOPs)** do que uma rede convencional, preservando a capacidade de extração de padrões morfológicos.

#### 4.4 Blocos Residuais Invertidos e Gargalo Linear (*Linear Bottlenecks*)
Conforme formalizado por **Sandler et al. (2018)**, os blocos intermediários da rede realizam uma expansão prévia de dimensionalidade (fator $t = 6$), processam as ativações no espaço expandido via convoluções *depthwise* e reprojetam o sinal de volta a um espaço linear comprimido (*bottleneck*). 

A função de ativação não linear (ReLU6) é propositalmente descartada na última projeção linear. Essa decisão evita o fenômeno de destruição da variedade de dados (*manifold destruction*), preservando detalhes sutis das lesões foliares, enquanto conexões residuais (*skip connections*) garantem a propagação suave dos gradientes.

---

### 5. RESULTADOS, MÉTRICAS E ANÁLISE CRÍTICA

#### 5.1 Formalização Matemática das Métricas de Avaliação
Respaldando-se na literatura canônica de classificação multiclasse (**Sokolova; Lapalme, 2009**), o desempenho do modelo é mensurado sobre a **Matriz de Confusão** $\mathbf{C} \in \mathbb{N}^{K \times K}$ (onde $K = 38$):
* **Acurácia Global (Top-1):** $\text{Acurácia} = \frac{\sum_{i=1}^{K} C_{i,i}}{\sum_{i=1}^{K} \sum_{j=1}^{K} C_{i,j}} = \frac{\sum_{i=1}^{K} \text{TP}_i}{N_{\text{total}}}$
* **Precisão por Classe ($\text{P}_i$):** $\text{P}_i = \frac{C_{i,i}}{\sum_{k=1}^{K} C_{k,i}} = \frac{\text{TP}_i}{\text{TP}_i + \text{FP}_i}$
* **Revocação por Classe ($\text{R}_i$):** $\text{R}_i = \frac{C_{i,i}}{\sum_{k=1}^{K} C_{i,k}} = \frac{\text{TP}_i}{\text{TP}_i + \text{FN}_i}$
* **F1-Score por Classe ($\text{F1}_i$):** $\text{F1}_i = 2 \cdot \frac{\text{P}_i \cdot \text{R}_i}{\text{P}_i + \text{R}_i}$ (Van Rijsbergen, 1979)
* **Macro F1-Score:** $\text{Macro F1} = \frac{1}{K} \sum_{i=1}^{K} \text{F1}_i$ (ponderação uniforme entre classes)
* **Weighted F1-Score:** $\text{Weighted F1} = \sum_{i=1}^{K} \left(\frac{N_i}{N_{\text{total}}}\right) \cdot \text{F1}_i$

#### 5.2 O Paradoxo da Acurácia em Conjuntos Desbalanceados
Conforme apontado por **He e Garcia (2009)**, conjuntos com disparidade demográfica acentuada tornam a Acurácia Global uma métrica enganosa (*Accuracy Paradox*). Um algoritmo que priorize classes dominantes pode alcançar índices globais elevados mesmo falhando na detecção de patologias raras. No contexto agronômico da AgroSense Analytics, deixar de alertar sobre um foco infeccioso incipiente (Falso Negativo) acarreta severo prejuízo à lavoura[cite: 4]. Por isso, a validação técnica exige a convergência simultânea entre a Acurácia Global (99,41%) e o **Macro F1-Score (99,19%)**.

#### 5.3 Procedimento de Avaliação Cega e Resultados de Teste ($N = 8.146$)
O teste final operou estritamente sobre as 8.146 amostras de teste inéditas por meio do script `src/evaluate_metrics.py`, adotando quatro critérios de rigor:
1. **Dados Inéditos:** avaliação exclusiva de folhas segregadas na partição inicial;
2. **Entrada Real Determinística:** desativação de perturbações sintéticas, mantendo apenas redimensionamento para $224 \times 224$ e normalização ImageNet[cite: 4, 7];
3. **Congelamento do Modelo:** execução em modo de inferência (`model.eval()`) sem rastreamento de gradientes (`torch.no_grad()`);
4. **Auditoria por Classe:** exportação automatizada das matrizes de contingência em `reports/metrics/classification_report.csv`.

| Indicador de Desempenho | Valor Aferido | Interpretação Técnica e Aplicação no Negócio |
| :--- | :---: | :--- |
| **Acurácia Global (Top-1)** | **99,41%** | 8.098 acertos exatos em 8.146 folhas de teste inéditas. |
| **Acurácia Top-3** | **99,96%** | Em 99,96% das inferências, o diagnóstico real está no top 3 retornado pelo app. |
| **Acurácia Top-5** | **99,98%** | Risco nulo de a doença ser totalmente omitida nas opções prioritárias. |
| **Macro F1-Score** | **99,19%** | Eficácia equilibrada entre classes raras e massivas, sem favorecer majoritárias[cite: 11]. |
| **Weighted F1-Score** | **99,41%** | Média harmônica ponderada pela representatividade real de cada cultura. |
| **Macro Recall (Sensibilidade)** | **99,38%** | Capacidade consistente de rastrear lesões em campo, minimizando falsos negativos. |
| **Macro Precision (Precisão)** | **99,07%** | Confiabilidade nos laudos emitidos, prevenindo aplicações desnecessárias de defensivos. |

#### 5.4 Auditoria da Classe Minoritária e Confusões Morfológicas
* **Auditoria da Classe `Potato___healthy`:** graças à função de perda `CrossEntropyLoss` ponderada com pesos inversamente proporcionais ao suporte amostral, a classe mais rara do acervo (suporte de teste $N=23$) obteve **100,00% de Revocação, 100,00% de Precisão e 100,00% de F1-Score**.
* **Confusões Morfológicas no Cultivo do Milho:** a menor precisão isolada ocorreu na classe *Corn Cercospora leaf spot* (83,70%), decorrente de confusões residuais com *Corn Northern Leaf Blight* (cuja revocação foi de 89,80%). Na fitopatologia agronômica, **Casela, Ferreira e Pinto (2006)** pela **Embrapa Milho e Sorgo** apontam que ambas as doenças produzem lesões necróticas lineares e retangulares muito próximas nos estágios iniciais, explicando a proximidade dos tensores convolucionais.
* **Mitigação de *Shortcut Learning*:** a inserção de perturbações de saturação e rotações preveniu que a rede associasse fundos de laboratório aos laudos, blindando os extratores contra o aprendizado de atalhos (*Shortcut Learning*, Geirhos et al., 2020).

---

### 6. CATÁLOGO DE FIGURAS E EVIDÊNCIAS VISUAIS

#### 6.1 Figura 5 — Curvas de Convergência de Treinamento e Validação
*(Inserir arquivo: `reports/figures/figura5_curvas_convergencia.png`)*  
**Fonte:** Desenvolvido pelos autores (2026).

A análise temporal das curvas de aprendizado da Figura 5 destaca:
* **Impacto do Transfer Learning:** convergência imediata já na primeira época, alcançando 97,0% de acurácia de validação com perda reduzida (0,12);
* **Efeito das Transformações Estocásticas:** no início do treinamento, a perda de treino apresenta-se superior à de validação em razão do ruído intencional injetado pelo *ColorJitter* e rotações estocásticas;
* **Estabilidade e Ausência de Overfitting:** as curvas convergem de modo suave e estável, sem que a curva de validação volte a subir nas iterações finais;
* **Critério de Parada Ótimo:** o melhor *checkpoint* foi registrado na **Época 5**, apresentando perda de validação de **0,0189** e acurácia de validação de **99,42%**, persistido em `models/mobilenetv2_plantvillage.pth`.

#### 6.2 Figura 6 — Matriz de Confusão Multiclasse Normalizada
*(Inserir arquivo: `reports/figures/figura6_matriz_confusao.png`)*  
**Fonte:** Desenvolvido pelos autores (2026).

O mapa de calor da Matriz de Confusão normalizada por revocação (*recall*) evidencia:
* **Diagonal Principal Dominante:** preservação de valores próximos a 1,00 em quase a totalidade das 38 patologias avaliadas;
* **Blindagem Inter-espécies:** total ausência de confusões grosseiras entre culturas distintas (folhas de tomate jamais foram confundidas com macieira ou videira);
* **Dispersão Residual Delimitada:** o único ponto perceptível fora da diagonal principal concentra-se estritamente na confusão mútua entre Cercosporiose e Helmintosporiose no milho, alinhado ao diagnóstico da literatura agronômica.

#### 6.3 Figura 7 — Galeria de Inferência Visual e Diagnóstico de Campo
*(Inserir arquivo: `reports/figures/figura7_galeria_diagnostico.jpg`)*  
**Fonte:** Desenvolvido pelos autores (2026).

A visualização de 16 inferências sobre amostras de teste inéditas demonstra o comportamento prático do produto:
* **Acerto Consistente em Diferentes Espécies:** classificação correta em 100% das amostras exibidas (`[CORRETO]`), transitando perfeitamente entre cultivos anuais e perenes com morfologias foliares distintas (tomate, soja, citros, videira, pimentão e pessegueiro);
* **Margem de Certeza Expressiva:** níveis de probabilidade *Softmax* situados entre **99,3% e 100,0%**, atestando separação clara no espaço vetorial latente;
* **Segurança na Identificação de Plantas Sadias:** discriminação confiável de amostras saudáveis (entre 99,7% e 100,0%), essencial para evitar aplicações desnecessárias de defensivos na lavoura[cite: 4];
* **Indiferença a Sombras e Variações de Fundo:** capacidade de diagnóstico preservada sob fundos com diferentes contrastes e sombras duras causadas pela curvatura da própria folha, comprovando foco nas lesões biológicas e imunidade a artefatos de fundo.

---

### 7. REFERÊNCIAS BIBLIOGRÁFICAS

CASELA, Carlos Roberto; FERREIRA, Alexandre da Silva; PINTO, Nicésio Firmino Geraldo de Alencar. **Doenças na Cultura do Milho**. Sete Lagoas: Embrapa Milho e Sorgo, 2006. 14 p. (Embrapa Milho e Sorgo. Circular Técnica, 83). Disponível em: <https://www.infoteca.cnptia.embrapa.br/bitstream/doc/490415/1/Circ83.pdf>.

GEIRHOS, Robert et al. **Shortcut learning in deep neural networks**. *Nature Machine Intelligence*, v. 2, n. 11, p. 665-673, nov. 2020. DOI: 10.1038/s42256-020-00257-z.

HE, Haibo; GARCIA, Edwardo A. **Learning from Imbalanced Data**. *IEEE Transactions on Knowledge and Data Engineering*, v. 21, n. 9, p. 1263-1284, set. 2009. DOI: 10.1109/TKDE.2008.239.

HOWARD, Andrew G. et al. **MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications**. *arXiv preprint arXiv:1704.04861*, 2017. Disponível em: <https://arxiv.org/abs/1704.04861>.

SANDLER, Mark et al. **MobileNetV2: Inverted Residuals and Linear Bottlenecks**. In: PROCEEDINGS OF THE IEEE CONFERENCE ON COMPUTER VISION AND PATTERN RECOGNITION (CVPR), 2018, Salt Lake City. Anais [...]. Piscataway: IEEE, 2018. p. 4510-4520. DOI: 10.1109/CVPR.2018.00474.

SOKOLOVA, Marina; LAPALME, Guy. **A systematic analysis of performance measures for classification tasks**. *Information Processing & Management*, v. 45, n. 4, p. 427-437, jul. 2009. DOI: 10.1016/j.ipm.2009.03.002.

VAN RIJSBERGEN, C. J. **Information Retrieval**. 2. ed. Londres: Butterworths, 1979.