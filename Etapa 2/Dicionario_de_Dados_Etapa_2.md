# AgroSense Analytics — Etapa 2: Produto Analítico
## Dicionário de Dados e Estrutura Tabular do Acervo de Imagens

**Projeto:** AgroSense Analytics — Detecção Automatizada de Fitopatologias por Visão Computacional  
**Componente Curricular:** Projeto Aplicado II  
**Autores:** Henrique Sarmento, Felipe Magalhães de Melo Costa, Jefferson Candido Moreira, Karla Maria Ramos da Silva  
**Base de Dados:** PlantVillage Dataset (Subconjunto `color` - RGB)  

---

### 1. Contextualização e Modelagem Tabular para Visão Computacional

Em projetos tradicionais de Ciência de Dados voltados para visão computacional, os dados brutos consistem em uma hierarquia de diretórios contendo matrizes numéricas tridimensionais (canais de cor $\times$ altura $\times$ largura). Para viabilizar a **Análise Exploratória de Dados (EDA)**, a garantia de integridade, o balanceamento de classes e a futura modelagem preditiva, foi construído um pipeline em Python que converte o acervo de imagens em um **DataFrame estruturado de metadados e atributos cromáticos**.

Este catálogo consolida tanto informações de negócio (cultura agrícola, condição fitossanitária e rótulo unificado) quanto variáveis técnicas da computação gráfica (dimensões espaciais, peso em disco, espaço de cores e estatísticas descritivas dos canais de cor RGB).

---

### 2. Dicionário de Dados Estruturado

A tabela a seguir apresenta a especificação detalhada das colunas que compõem o DataFrame do projeto:

| Coluna | Descrição | Tipo de Dado | Exemplo de Valor |
| :--- | :--- | :--- | :--- |
| `caminho_arquivo` | Caminho relativo do arquivo de imagem a partir da raiz do projeto | String (Texto) | `plantvillage dataset/color/Tomato___Target_Spot/0a0b1c2d.jpg` |
| `nome_arquivo` | Nome do arquivo físico com a respectiva extensão | String (Texto) | `0a0b1c2d.jpg` |
| `especie_cultura` | Espécie da cultura agrícola à qual a folha pertence (ex: Tomate, Maçã, Milho, Batata) | String (Texto) | `Tomato`, `Apple`, `Corn_(maize)` |
| `condicao_sanitaria` | Diagnóstico fitopatológico ou estado de higidez da folha cultivada | String (Texto) | `Target_Spot`, `healthy`, `Late_blight` |
| `classe_rotulo` | Identificador unificado da classe (combinação da espécie vegetal e patologia) | String (Texto) | `Tomato___Target_Spot` |
| `is_saudavel` | Indicador binário que sinaliza se o espécime está saudável (`True`) ou infectado (`False`) | Booleano | `True` / `False` |
| `largura_px` | Dimensão horizontal da matriz de imagem em pixels | Inteiro (`int64`) | `256` |
| `altura_px` | Dimensão vertical da matriz de imagem em pixels | Inteiro (`int64`) | `256` |
| `total_pixels` | Quantidade total de posições amostrais na grade espacial ($largura \times altura$) | Inteiro (`int64`) | `65536` |
| `qtd_canais` | Número de camadas matriciais de cores presentes na imagem | Inteiro (`int64`) | `3` (RGB) |
| `espaco_cores` | Sistema de representação cromática das imagens | String (Texto) | `RGB` |
| `tamanho_arquivo_kb` | Tamanho do arquivo compactado em disco expresso em kilobytes | Float (`float64`) | `14.85` |
| `formato_arquivo` | Formato do encapsulamento e compressão do arquivo digital | String (Texto) | `JPEG` |
| `media_canal_r` | Intensidade luminosa média dos pixels no canal Vermelho (escala de 0 a 255) | Float (`float64`) | `124.32` |
| `media_canal_g` | Intensidade luminosa média dos pixels no canal Verde (escala de 0 a 255) | Float (`float64`) | `145.18` |
| `media_canal_b` | Intensidade luminosa média dos pixels no canal Azul (escala de 0 a 255) | Float (`float64`) | `102.40` |
| `desvio_canal_r` | Dispersão (desvio padrão) dos níveis de cinza no canal Vermelho | Float (`float64`) | `42.15` |
| `desvio_canal_g` | Dispersão (desvio padrão) dos níveis de cinza no canal Verde | Float (`float64`) | `38.60` |
| `desvio_canal_b` | Dispersão (desvio padrão) dos níveis de cinza no canal Azul | Float (`float64`) | `45.92` |

---

### 3. Justificativa Técnica das Variáveis

1. **Variáveis Categóricas e Alvo (`especie_cultura`, `condicao_sanitaria`, `classe_rotulo`, `is_saudavel`):**
   - Permitem tanto abordagens de classificação multiclasse (38 categorias) quanto classificação binária hierárquica (Saudável vs. Doente).
   - A coluna `is_saudavel` viabiliza uma triagem primária no aplicativo mobile antes da identificação do patógeno específico.

2. **Variáveis Físicas e Geométricas (`largura_px`, `altura_px`, `total_pixels`, `qtd_canais`, `espaco_cores`, `formato_arquivo`):**
   - Garantem que a base atenda ao pré-requisito de homogeneidade dimensional exigido por redes neurais convolucionais (CNNs), eliminando distorções de aspecto ou necessidade de padding assimétrico.

3. **Métricas Fotométricas e Estatísticas Cromáticas (`media_canal_*`, `desvio_canal_*`):**
   - Fornecem subsídios para detectar anomalias cromáticas causadas por necroses, cloroses ou ferrugens (que tipicamente diminuem a resposta espectral no verde e aumentam reflexões amarelas/castanhas).
   - Servem como base de cálculo direta para as etapas de normalização e padronização (Z-score ou Min-Max) durante o treinamento dos modelos na Etapa 3.

---

### 4. Instruções de Integração no Relatório Acadêmico (Word)

Esta seção foi redigida para atender diretamente aos itens da **Etapa 2 - Definição do Produto Analítico**:
- Pode ser copiada diretamente para a subseção de **Dicionário de Dados e Análise Exploratória** do arquivo `Modelo de entrega.docx`.
- Os códigos geradores e análises visuais complementares estão disponíveis no notebook `ProjetoAplicado2/notebooks/Etapa_2_Analise_Exploratoria_e_Dicionario.ipynb`.
