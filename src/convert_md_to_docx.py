"""
AgroSense Analytics — Conversor do Relatorio Academico da Etapa 2 para Word (.docx)
Segue estritamente as normas ABNT:
- Margens: Sup/Esq: 3,0 cm | Inf/Dir: 2,0 cm
- Fonte: Arial 12 pt, entrelinhas 1,5 pt, recuo 1,25 cm nos paragrafos
- Tabelas nativas estilizadas com bordas discretas e cabecalho sombreado
- Incorporacao centralizada das 7 figuras em 300 DPI com legendas em 10 pt
- Formulas matematicas destacadas e alinhadas
"""

import os
import sys
from pathlib import Path

import docx
from docx.shared import Cm, Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn


def set_cell_background(cell, fill_hex):
    """Aplica cor de preenchimento de fundo em uma celula da tabela."""
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shd)


def set_cell_margins(cell, top=120, bottom=120, left=160, right=160):
    """Define margens internas (padding) de uma celula (unidades em dxa)."""
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    cell._tc.get_or_add_tcPr().append(tcMar)


def set_table_borders(table, color="B0BEC5", sz="4"):
    """Aplica bordas horizontais discretas e remove bordas verticais para elegancia ABNT."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideH w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)


def format_paragraph(p, first_line_indent=1.25, line_spacing=1.5, space_after=6, space_before=0, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    """Configura o formato de paragrafo compativel com a norma ABNT."""
    p.alignment = align
    pf = p.paragraph_format
    pf.first_line_indent = Cm(first_line_indent)
    pf.line_spacing = line_spacing
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)


def add_custom_heading1(doc, text):
    """Adiciona titulo de Secao Nivel 1 (ex: 1. Secao 1: ...) em Arial 14 pt Negrito."""
    p = doc.add_paragraph()
    format_paragraph(p, first_line_indent=0, line_spacing=1.5, space_before=16, space_after=6, align=WD_ALIGN_PARAGRAPH.LEFT)
    run = p.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(14)
    run.font.bold = True
    run.font.color.rgb = RGBColor(26, 35, 126)  # Azul escuro institucional
    return p


def add_custom_heading2(doc, text):
    """Adiciona titulo de Subsecao Nivel 2 (ex: 1.1 ...) em Arial 12 pt Negrito."""
    p = doc.add_paragraph()
    format_paragraph(p, first_line_indent=0, line_spacing=1.5, space_before=12, space_after=4, align=WD_ALIGN_PARAGRAPH.LEFT)
    run = p.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(12)
    run.font.bold = True
    run.font.color.rgb = RGBColor(38, 50, 56)
    return p


def add_custom_heading3(doc, text):
    """Adiciona titulo de Nivel 3 em Arial 12 pt Negrito/Italico."""
    p = doc.add_paragraph()
    format_paragraph(p, first_line_indent=0, line_spacing=1.5, space_before=8, space_after=3, align=WD_ALIGN_PARAGRAPH.LEFT)
    run = p.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(12)
    run.font.bold = True
    run.font.italic = True
    run.font.color.rgb = RGBColor(55, 71, 79)
    return p


def add_body_paragraph(doc, text):
    """Adiciona paragrafo padrao de corpo de texto em Arial 12 pt justificado."""
    p = doc.add_paragraph()
    format_paragraph(p, first_line_indent=1.25, line_spacing=1.5, space_after=6, space_before=0, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    run = p.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor(0, 0, 0)
    return p


def add_bullet_item(doc, bold_prefix, text):
    """Adiciona item com marcador em lista (bullet point)."""
    p = doc.add_paragraph(style='List Bullet')
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.first_line_indent = Cm(-0.5)
    pf.left_indent = Cm(1.25)
    pf.line_spacing = 1.3
    pf.space_after = Pt(4)
    pf.space_before = Pt(2)
    
    run_b = p.add_run(bold_prefix)
    run_b.font.name = 'Arial'
    run_b.font.size = Pt(11.5)
    run_b.font.bold = True
    run_b.font.color.rgb = RGBColor(0, 0, 0)
    
    run_t = p.add_run(text)
    run_t.font.name = 'Arial'
    run_t.font.size = Pt(11.5)
    run_t.font.color.rgb = RGBColor(0, 0, 0)
    return p


def add_formula_block(doc, formula_text, label=""):
    """Adiciona bloco destacado para formulas matematicas centralizadas."""
    p = doc.add_paragraph()
    format_paragraph(p, first_line_indent=0, line_spacing=1.2, space_before=6, space_after=6, align=WD_ALIGN_PARAGRAPH.CENTER)
    
    run = p.add_run(formula_text)
    run.font.name = 'Cambria Math'
    run.font.size = Pt(11.5)
    run.font.italic = True
    run.font.bold = True
    run.font.color.rgb = RGBColor(13, 71, 161)
    
    if label:
        run_l = p.add_run(f"    ({label})")
        run_l.font.name = 'Arial'
        run_l.font.size = Pt(10)
        run_l.font.italic = False
        run_l.font.color.rgb = RGBColor(100, 100, 100)
    return p


def add_figure_with_caption(doc, image_path, caption_text, width=Cm(14.0)):
    """Insere imagem centralizada com largura padronizada e legenda conforme normas ABNT."""
    p_img = doc.add_paragraph()
    format_paragraph(p_img, first_line_indent=0, line_spacing=1.0, space_before=10, space_after=4, align=WD_ALIGN_PARAGRAPH.CENTER)
    
    run_img = p_img.add_run()
    run_img.add_picture(str(image_path), width=width)
    
    p_cap = doc.add_paragraph()
    format_paragraph(p_cap, first_line_indent=0, line_spacing=1.1, space_before=2, space_after=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    
    run_cap = p_cap.add_run(caption_text)
    run_cap.font.name = 'Arial'
    run_cap.font.size = Pt(10)
    run_cap.font.bold = True
    run_cap.font.color.rgb = RGBColor(38, 50, 56)
    return p_img, p_cap


def create_styled_table(doc, headers, rows_data, col_widths=None):
    """Constroi tabela nativa do Word com formatacao visual profissional e bordas discretas."""
    tbl = doc.add_table(rows=len(rows_data) + 1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl, color="B0BEC5", sz="4")

    # Linha de Cabecalho
    hdr_cells = tbl.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], "E8ECEF")
        set_cell_margins(hdr_cells[i], top=140, bottom=140, left=180, right=180)
        p = hdr_cells[i].paragraphs[0]
        format_paragraph(p, first_line_indent=0, line_spacing=1.1, space_after=2, space_before=2, align=WD_ALIGN_PARAGRAPH.CENTER)
        for r in p.runs:
            r.font.name = 'Arial'
            r.font.size = Pt(10)
            r.font.bold = True
            r.font.color.rgb = RGBColor(26, 35, 126)

    # Linhas de Dados
    for r_idx, row in enumerate(rows_data):
        row_cells = tbl.rows[r_idx + 1].cells
        bg_color = "F9FAFB" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], top=100, bottom=100, left=160, right=160)
            p = row_cells[c_idx].paragraphs[0]
            align = WD_ALIGN_PARAGRAPH.CENTER if (c_idx > 0 and len(val) < 15) else WD_ALIGN_PARAGRAPH.LEFT
            format_paragraph(p, first_line_indent=0, line_spacing=1.15, space_after=2, space_before=2, align=align)
            for r in p.runs:
                r.font.name = 'Arial'
                r.font.size = Pt(9.5)
                r.font.color.rgb = RGBColor(33, 33, 33)

    # Ajuste de largura de colunas
    if col_widths:
        for r in tbl.rows:
            for c_idx, w in enumerate(col_widths):
                r.cells[c_idx].width = w

    # Espacamento apos a tabela
    p_spacer = doc.add_paragraph()
    format_paragraph(p_spacer, first_line_indent=0, line_spacing=1.0, space_after=6, space_before=2)
    return tbl


def build_document():
    doc = docx.Document()

    # 1. Configuracao das Margens ABNT
    section = doc.sections[0]
    section.top_margin = Cm(3.0)
    section.left_margin = Cm(3.0)
    section.bottom_margin = Cm(2.0)
    section.right_margin = Cm(2.0)

    # 2. Cabecalho Institucional / Capa Integrada
    p_top = doc.add_paragraph()
    format_paragraph(p_top, first_line_indent=0, line_spacing=1.3, space_before=0, space_after=4, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_inst = p_top.add_run("PROJETO APLICADO II — CIÊNCIA DE DADOS\n")
    r_inst.font.name = 'Arial'
    r_inst.font.size = Pt(13)
    r_inst.font.bold = True
    r_inst.font.color.rgb = RGBColor(100, 100, 100)

    r_title = p_top.add_run("ETAPA 2: DEFINIÇÃO DO PRODUTO ANALÍTICO\n")
    r_title.font.name = 'Arial'
    r_title.font.size = Pt(16)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(26, 35, 126)

    r_sub = p_top.add_run("Classificação Automatizada de Fitopatologias com MobileNetV2 e Edge AI no Agronegócio")
    r_sub.font.name = 'Arial'
    r_sub.font.size = Pt(12)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(60, 60, 60)

    # Box de Dados Institucionais
    p_box = doc.add_paragraph()
    format_paragraph(p_box, first_line_indent=0, line_spacing=1.2, space_before=10, space_after=14, align=WD_ALIGN_PARAGRAPH.LEFT)
    box_runs = [
        ("Startup: ", True), ("AgroSense Analytics\n", False),
        ("Autores: ", True), ("Henrique Sarmento, Felipe Magalhães de Melo Costa, Jefferson Candido Moreira, Karla Maria Ramos da Silva\n", False),
        ("Repositório Oficial no GitHub: ", True), ("https://github.com/henri-sarmento/ProjetoAplicado2\n", False),
        ("Base de Dados Experimental: ", True), ("PlantVillage Dataset (Subconjunto color — 54.305 imagens RGB, 38 classes)", False)
    ]
    for text, bold in box_runs:
        r = p_box.add_run(text)
        r.font.name = 'Arial'
        r.font.size = Pt(10.5)
        r.font.bold = bold
        r.font.color.rgb = RGBColor(33, 33, 33)

    # Separador
    p_sep = doc.add_paragraph()
    format_paragraph(p_sep, first_line_indent=0, line_spacing=1.0, space_before=0, space_after=12)
    r_sep = p_sep.add_run("―" * 65)
    r_sep.font.color.rgb = RGBColor(180, 180, 180)

    # =========================================================================
    # SEÇÃO 1: BIBLIOTECAS E REPOSITÓRIO
    # =========================================================================
    add_custom_heading1(doc, "1. Seção 1: Bibliotecas e Repositório")

    add_custom_heading2(doc, "1.1 Relação Formal do Ecossistema Tecnológico")
    add_body_paragraph(doc, 
        "Para a concepção, exploração, pré-processamento, treinamento e avaliação do produto analítico da "
        "AgroSense Analytics, foi estabelecido um ambiente computacional padronizado na linguagem Python (versão 3.12). "
        "O projeto adota uma arquitetura modular de desenvolvimento suportada por bibliotecas consagradas pela comunidade "
        "científica de Aprendizado Profundo e Visão Computacional, garantindo reprodutibilidade estrita e alto desempenho.")

    # Tabela 1: Bibliotecas
    tbl1_headers = ["Biblioteca / Pacote", "Versão Utilizada", "Função Primária no Pipeline Analítico da AgroSense"]
    tbl1_data = [
        ["PyTorch (torch)", "2.6.0+cu124", "Grafo computacional dinâmico, Autograd, treinamento acelerado em GPU e gestão de tensores."],
        ["Torchvision", "0.21.0+cu124", "Topologia MobileNetV2 pré-treinada (Transfer Learning), Data Augmentation e normalização de imagem."],
        ["Scikit-Learn", "1.9.1", "Particionamento estratificado 70/15/15, classification_report, acurácia e matriz de confusão."],
        ["Pandas", "3.0.5", "Construção, indexação e auditoria tabular do Dicionário de Metadados do acervo."],
        ["NumPy", "2.5.2", "Operações matriciais de baixo nível, vetorização e estatísticas fotométricas."],
        ["Pillow (PIL)", "12.3.0", "Decodificação defensiva de arquivos de imagem e conversão estrita para o espaço RGB."],
        ["Matplotlib / Seaborn", "3.11.1 / 0.13.2", "Geração e exportação dos gráficos de EDA, curvas de treino e matriz de confusão a 300 DPI."]
    ]
    create_styled_table(doc, tbl1_headers, tbl1_data, [Cm(4.0), Cm(3.0), Cm(9.0)])

    add_custom_heading2(doc, "1.2 Repositório Oficial do Projeto")
    add_body_paragraph(doc,
        "O versionamento, controle de branches e governança do código-fonte estão centralizados no GitHub sob a URL oficial: "
        "https://github.com/henri-sarmento/ProjetoAplicado2. A estrutura do repositório foi construída seguindo o padrão de "
        "engenharia de dados com segregação de pastas entre código-fonte (src/), dados (data/), notebooks (notebooks/), "
        "pesos binários (models/) e artefatos de saída gráfica e métrica (reports/figures/ e reports/metrics/).")

    add_custom_heading2(doc, "1.3 Justificativa Técnica da Escolha do Ecossistema PyTorch com Aceleração CUDA")
    add_body_paragraph(doc,
        "A opção estratégica pelo ecossistema PyTorch compilado com suporte nativo a CUDA 12.4 — em detrimento de abordagens "
        "baseadas exclusivamente em CPU ou frameworks de alto nível fechados (como pipelines encapsulados de AutoML ou "
        "interfaces proprietárias) — fundamenta-se em três pilares técnicos essenciais:")

    add_bullet_item(doc, "Aceleração por Hardware Dedicado (NVIDIA Tensor Cores): ",
        "O treinamento de redes convolucionais profundas envolve milhões de operações de multiplicação de matrizes e acumulação "
        "(GEMM). A utilização da GPU NVIDIA GeForce RTX 3060 (12 GB de VRAM GDDR6) via CUDA 12.4 permitiu alocar lotes (mini-batches) "
        "de 64 imagens concomitantemente, viabilizando o processamento das 54.305 amostras em apenas 19,24 minutos (cerca de 228 segundos "
        "por época). Na CPU, o mesmo processo exigiria horas de processamento contínuo com severo gargalo térmico e de latência.")

    add_bullet_item(doc, "Grafo Computacional Dinâmico e Depurabilidade Pythonica: ",
        "Diferente de grafos estáticos fechados, o PyTorch constrói o grafo de diferenciação automática em tempo de execução (define-by-run). "
        "Isso possibilita inspeção direta de gradientes, implementação flexível de penalizações por classe na função de perda "
        "(CrossEntropyLoss ponderada) e manipulação customizada do fluxo de tensores.")

    add_bullet_item(doc, "Alinhamento com o Modelo de Negócio da AgroSense Analytics (Edge AI): ",
        "Como a proposta de valor da startup prevê a execução em dispositivos móveis no campo (Android), o PyTorch oferece caminhos nativos "
        "de exportação e otimização para inferência em borda, como TorchScript, quantização int8 e conversão para ONNX (Open Neural Network Exchange), "
        "permitindo que a inteligência artificial embarcada rode localmente no smartphone do agricultor, inclusive em áreas rurais desprovidas de internet.")

    # =========================================================================
    # SEÇÃO 2: ANÁLISE EXPLORATÓRIA (EDA) E DICIONÁRIO DE METADADOS
    # =========================================================================
    add_custom_heading1(doc, "2. Seção 2: Análise Exploratória (EDA) e Dicionário de Metadados")

    add_custom_heading2(doc, "2.1 Visão Geral do Acervo PlantVillage")
    add_body_paragraph(doc,
        "O repositório PlantVillage adotado como base analítica da AgroSense Analytics é composto por um total de 54.305 imagens digitais "
        "exclusivas, segmentadas em 38 classes fitopatológicas. O acervo abrange 14 espécies vegetais distintas (como tomate, batata, maçã, "
        "milho, uva, pimentão e soja) distribuídas entre espécimes completamente saudáveis e folhas infectadas por patógenos fúngicos, "
        "bacterianos, virais e ácaros. Todas as imagens do subconjunto colorimétrico (color) encontram-se padronizadas na resolução espacial "
        "de 256 x 256 pixels, codificadas no formato de compressão JPEG.")

    add_custom_heading2(doc, "2.2 Dicionário de Metadados Estruturado")
    add_body_paragraph(doc,
        "Para estruturar o catálogo do acervo físico e viabilizar a auditoria estatística do projeto, foi desenvolvido o script "
        "src/extract_metadata.py. O pipeline percorreu concorrentemente as 54.305 imagens e gerou a base relacional "
        "data/metadados_plantvillage.csv, detalhada no dicionário de dados a seguir:")

    # Tabela 2: Dicionario de Metadados
    tbl2_headers = ["Coluna", "Descrição Técnica e de Negócio", "Tipo de Dado", "Exemplo de Valor"]
    tbl2_data = [
        ["caminho_arquivo", "Localizador relativo do arquivo de imagem a partir da raiz", "String", "plantvillage dataset/color/Tomato___Target_Spot/0a0b1c2d.jpg"],
        ["nome_arquivo", "Identificador físico do arquivo com extensão digital", "String", "0a0b1c2d-96e0-4a87-8d07-299f0e263d59___UNREST.JPG"],
        ["especie_cultura", "Espécie agrícola à qual pertence a folha cultivada", "String", "Tomato, Apple, Corn_(maize), Potato"],
        ["condicao_sanitaria", "Diagnóstico fitopatológico específico ou condição saudável", "String", "Target_Spot, healthy, Early_blight, Black_rot"],
        ["classe_rotulo", "Identificador unificado da classe (espécie + patologia)", "String", "Tomato___Target_Spot (38 categorias únicas)"],
        ["is_saudavel", "Flag booleana indicativa de higidez da folha", "Booleano", "True (Saudável) / False (Infectada)"],
        ["largura_px", "Dimensão horizontal da matriz de imagem", "Inteiro (int64)", "256"],
        ["altura_px", "Dimensão vertical da matriz de imagem", "Inteiro (int64)", "256"],
        ["total_pixels", "Produto espacial da grade de amostragem (largura * altura)", "Inteiro (int64)", "65536"],
        ["qtd_canais", "Profundidade da camada cromática matricial", "Inteiro (int64)", "3 (54.304 imagens) ou 4 (1 imagem RGBA)"],
        ["espaco_cores", "Modelo espectral de representação das cores", "String", "RGB (54.304 imagens) ou RGBA (1 imagem)"],
        ["tamanho_arquivo_kb", "Tamanho do arquivo digital compactado em disco", "Float (float64)", "14.85"],
        ["formato_arquivo", "Norma de compressão e encapsulamento digital", "String", "JPEG"],
        ["media_canal_r", "Intensidade média de reflexão no canal Vermelho (0–255)", "Float (float64)", "124.32"],
        ["media_canal_g", "Intensidade média de reflexão no canal Verde (0–255)", "Float (float64)", "145.18"],
        ["media_canal_b", "Intensidade média de reflexão no canal Azul (0–255)", "Float (float64)", "102.40"],
        ["desvio_canal_r", "Dispersão estatística das intensidades no canal Vermelho", "Float (float64)", "42.15"],
        ["desvio_canal_g", "Dispersão estatística das intensidades no canal Verde", "Float (float64)", "38.60"],
        ["desvio_canal_b", "Dispersão estatística das intensidades no canal Azul", "Float (float64)", "45.92"]
    ]
    create_styled_table(doc, tbl2_headers, tbl2_data, [Cm(3.5), Cm(6.5), Cm(2.5), Cm(3.5)])

    add_custom_heading2(doc, "2.3 Análise Visual e Exploração de Dados (EDA)")
    add_body_paragraph(doc,
        "A fundamentação analítica da etapa exploratória baseia-se em quatro figuras estruturadas exportadas em alta resolução (300 DPI) "
        "para subsidiar o relatório técnico, discutidas a seguir:")

    # Figura 1
    add_custom_heading3(doc, "A. Distribuição de Frequências e Diagnóstico de Desbalanceamento Severo")
    add_body_paragraph(doc,
        "A Figura 1 apresenta o histograma de frequências ordenado para todas as 38 classes fitopatológicas do acervo, destacando visualmente "
        "a disparidade entre as categorias extremas da base de dados:")
    
    fig1_path = Path("reports/figures/eda_distribuicao_classes.png")
    if fig1_path.exists():
        add_figure_with_caption(doc, fig1_path,
            "Figura 1 — Distribuição de Frequência das 38 Classes Fitopatológicas do Dataset PlantVillage.", width=Cm(14.0))

    add_body_paragraph(doc,
        "A inspeção empírica da Figura 1 revela uma disparidade demográfica substancial entre as categorias do dataset: a classe majoritária "
        "(Orange Citrus Greening) possui 5.507 imagens (10,14% do acervo total), enquanto a classe minoritária (Potato healthy) dispõe de apenas "
        "152 imagens (0,28% do acervo total). Essa configuração estabelece uma razão de desbalanceamento severa de 36,2 : 1. "
        "Sob a ótica de engenharia de aprendizado de máquina, essa constatação impôs duas decisões arquiteturais compulsórias na modelagem: "
        "(1) a adoção mandante de particionamento estratificado por classe e (2) a utilização de penalização de pesos ponderados inversos "
        "à frequência na função de perda (CrossEntropyLoss), impedindo que o gradiente do otimizador seja monopolizado pelas classes massivas.")

    # Figura 2
    add_custom_heading3(doc, "B. Censo Fitossanitário do Acervo")
    add_body_paragraph(doc,
        "A Figura 2 sintetiza o estado de higidez foliar agregado da base de dados, confrontando a proporção de espécimes doentes versus íntegros:")

    fig2_path = Path("reports/figures/eda_status_sanitario.png")
    if fig2_path.exists():
        add_figure_with_caption(doc, fig2_path,
            "Figura 2 — Proporção Geral entre Folhas Sadias e Espécimes Infectados.", width=Cm(14.0))

    add_body_paragraph(doc,
        "Os dados quantitativos revelam 39.221 imagens infectadas (72,22%) contra 15.084 imagens sadias (27,78%). Essa proporção reflete "
        "com precisão o contexto de negócio da AgroSense Analytics: na agricultura comercial, o monitoramento fitossanitário é predominantemente "
        "voltado à detecção precoce de anomalias patológicas, que constituem a ampla maioria dos espécimes sob suspeita clínica levados ao aplicativo mobile.")

    # Figura 3
    add_custom_heading3(doc, "C. Mosaico Morfológico de Sintomas Fitopatológicos")
    add_body_paragraph(doc,
        "A Figura 3 exibe uma grade comparativa de amostras reais em alta definição extraídas do dataset, ilustrando as manifestações "
        "fenotípicas das principais patologias:")

    fig3_path = Path("reports/figures/eda_mosaico_patologias.png")
    if fig3_path.exists():
        add_figure_with_caption(doc, fig3_path,
            "Figura 3 — Mosaico Amostral de Patologias Foliares: Sintomas Visuais de Necrose, Clorose e Ferrugem.", width=Cm(14.5))

    add_body_paragraph(doc,
        "A análise visual da Figura 3 comprova a rica diversidade de padrões sintomatológicos observáveis: necrose concêntrica em Tomato Target Spot, "
        "clorose e encarquilhamento foliar em Tomato TYLCV, pústulas eruptivas de ferrugem alaranjada em Corn Common Rust, micélio pulverulento "
        "branco de oídio em Squash Powdery Mildew, requeima severa de fitóftora em Potato Late Blight, podridão negra circular em Grape Black Rot e "
        "folhas íntegras homogêneas em Apple healthy e Pepper healthy. Essa diversidade morfológica comprova que os marcadores patológicos são puramente "
        "espaciais e geométricos, sendo invisíveis a resumos estatísticos tabulares.")

    # Figura 4
    add_custom_heading3(doc, "D. Assinatura Fotométrica e Dispersão Espectral RGB")
    add_body_paragraph(doc,
        "A Figura 4 traça as curvas contínuas de densidade de probabilidade (KDE) e histogramas das intensidades médias de pixel para os "
        "canais Vermelho (R), Verde (G) e Azul (B), comparando folhas sadias e infectadas:")

    fig4_path = Path("reports/figures/eda_distribuicao_rgb.png")
    if fig4_path.exists():
        add_figure_with_caption(doc, fig4_path,
            "Figura 4 — Distribuição de Densidade das Intensidades Médias de Pixel nos Canais R, G e B.", width=Cm(14.5))

    add_body_paragraph(doc,
        "A curva fotométrica explicita a fisiologia vegetal: folhas sadias exibem pico de reflectância em maiores intensidades de verde (media ~ 148,2), "
        "condizente com a alta concentração de clorofila funcional que reflete na faixa de 500–550 nm. Folhas infectadas manifestam cauda alongada "
        "nas intensidades médias de vermelho (media ~ 126,5), provocada por tecidos necrosados e desidratados. Embora haja desvios médios discerníveis, "
        "as curvas apresentam considerável sobreposição espectral, comprovando que a média de cor isolada não possui poder discriminante suficiente, "
        "exigindo o poder de extração morfológica de redes convolucionais.")

    add_custom_heading2(doc, "2.4 Justificativa Conceitual Crítica: Inviabilidade de Modelagem Preditiva Tabular")
    add_body_paragraph(doc,
        "Um dos pontos de maior rigor científico da Etapa 2 consiste em responder: Por que a tabela metadados_plantvillage.csv não pôde ser "
        "utilizada como entrada de um modelo preditivo tradicional (como Random Forest, XGBoost ou Regressão Logística tabular)? "
        "A fundamentação analítica sustenta-se em duas barreiras epistemológicas intrínsecas ao problema:")

    add_bullet_item(doc, "Vazamento Direto de Rótulo (Target Leakage Estrutural): ",
        "As colunas especie_cultura, condicao_sanitaria e is_saudavel foram obtidas por decomposição léxica da pasta de origem no disco "
        "(classe_rotulo.split('___')). Em particular, is_saudavel = II(condicao_sanitaria == 'healthy'). Se essas variáveis fossem "
        "alimentadas a um algoritmo tabular preditivo, o modelo aprenderia uma correlação perfeita, trivial e espúria, obtendo acurácia "
        "artificial de 100% sem jamais examinar o vegetal. Na prática de campo da AgroSense Analytics, quando um produtor aponta a câmera "
        "do celular para uma folha doente, o sistema não recebe metadados textuais sobre o diagnóstico — a imagem é o único sinal observável.")

    add_bullet_item(doc, "Natureza da Representação Espacial das Fitopatologias: ",
        "Os marcadores biológicos que caracterizam as 38 fitopatologias não residem em resumos escalares de arquivos digitais (peso em KB, "
        "largura em pixels ou média cromática global). Os sinais fisiológicos diagnósticos dependem estritamente da topologia espacial e "
        "textura local da matriz de pixels tridimensional X em R^(3 x 224 x 224). Cloroses manifestam-se em halos concêntricos e manchas "
        "angulares internervais; pústulas de ferrugem exigem detecção de contraste pontual; manchas alvo requerem círculos concêntricos. "
        "Variáveis como media_canal_r condensam 65.536 pixels em um único escalar, destruindo toda a correlação de vizinhança espacial entre "
        "pixels adjacentes. Dessa forma, métodos tabulares baseados em metadados são cegos à morfologia patológica vegetal, sendo imperativa "
        "a adoção de Redes Neurais Convolucionais (CNNs) operando diretamente na grade espacial contínua de pixels.")

    # =========================================================================
    # SEÇÃO 3: TRATAMENTO DE DADOS E ESTRATÉGIA DE PARTICIONAMENTO
    # =========================================================================
    add_custom_heading1(doc, "3. Seção 3: Tratamento de Dados e Estratégia de Particionamento")

    add_custom_heading2(doc, "3.1 Particionamento Estratificado (70% Treino, 15% Validação e 15% Teste)")
    add_body_paragraph(doc,
        "O particionamento amostral foi formulado para garantir a integridade estatística do aprendizado supervisionado, dividindo o conjunto "
        "total (N = 54.305) da seguinte forma:")

    # Tabela 3: Splits
    tbl3_headers = ["Subconjunto de Dados", "Proporção", "Volume Absoluto", "Função Metodológica no Pipeline da AgroSense"]
    tbl3_data = [
        ["Treinamento (df_train)", "70,0%", "38.013 amostras", "Otimização dos pesos sinápticos via retropropagação de gradientes (AdamW)."],
        ["Validação (df_val)", "15,0%", "8.146 amostras", "Ajuste fino de hiperparâmetros, scheduler adaptativo e seleção do melhor checkpoint."],
        ["Teste Inédito (df_test)", "15,0%", "8.146 amostras", "Avaliação final cega e imparcial da capacidade de generalização do produto analítico."],
        ["Total Consolidado", "100,0%", "54.305 amostras", "Representatividade integral das 38 classes em todas as três partições."]
    ]
    create_styled_table(doc, tbl3_headers, tbl3_data, [Cm(4.0), Cm(2.5), Cm(3.5), Cm(6.0)])

    add_body_paragraph(doc,
        "Conforme diagnosticado na Seção 2, a disparidade demográfica de 36,2 : 1 entre as classes tornaria desastrosa uma amostragem aleatória "
        "simples, na qual a probabilidade de selecionar k amostras da classe Potato healthy seguiria uma distribuição binomial com risco iminente "
        "de sub-representação extrema ou aniquilação nos conjuntos de teste. A estratificação forçou matematicamente a preservação exata da "
        "distribuição marginal de probabilidade P(Y = c) em todas as três partições, assegurando que todas as 38 classes estivessem presentes "
        "de forma fidedigna no treino, na validação e no teste.")

    add_body_paragraph(doc,
        "Além disso, a integridade das partições foi validada matematicamente através de asserções estritas de teoria dos conjuntos sobre os "
        "índices de amostragem: a interseção entre os conjuntos de índices de Treino, Validação e Teste é rigorosamente vazia, e a união dos três "
        "equivale ao acervo total. Isso garante matematicamente a ausência completa de vazamento de dados (Zero Data Leakage).")

    add_custom_heading2(doc, "3.2 Pipeline de Pré-Processamento e Transformações Estocásticas")
    add_body_paragraph(doc,
        "O pipeline de pré-processamento das imagens implementou decisões rigorosas de engenharia:")

    add_bullet_item(doc, "Normalização Pré-Calculada do ImageNet vs. Fit no Dataset: ",
        "A normalização dos tensores de imagem foi aplicada através da transformação afim x' = (x - mu) / sigma, utilizando os vetores pré-computados "
        "do ImageNet: mu = [0.485, 0.456, 0.406] e sigma = [0.229, 0.224, 0.225]. Em Transfer Learning com pesos da MobileNetV2, os núcleos "
        "convolucionais pré-treinados foram otimizados exatamente no espaço de distribuições do ImageNet (1,4 milhão de imagens). Recalcular as "
        "médias estatísticas sobre o dataset PlantVillage inteiro geraria Data Leakage conceitual caso feito antes do split e induziria Covariate "
        "Shift artificial na entrada dos filtros convolucionais, forçando o regradiente das primeiras camadas e retardando a convergência.")

    add_bullet_item(doc, "Confinamento Estrito do Data Augmentation ao Treinamento: ",
        "No conjunto de treinamento (train_transforms), foram aplicadas transformações aleatórias estocásticas em tempo de execução: redimensionamento "
        "para 224 x 224 pixels, espelhamento horizontal (p = 0.5), espelhamento vertical (p = 0.2), rotação aleatória contínua em até 20 graus e "
        "ColorJitter variando brilho, contraste e saturação em 15% e matiz em 5%. Essa expansão sintética funciona como uma prior Bayesiana de "
        "invariância de iluminação e rotação, prevenindo overfitting. Nos conjuntos de validação e teste, aplicou-se estritamente redimensionamento "
        "determinístico e normalização, garantindo que o teste reflita a morfologia fidedigna das imagens sem distorções artificiais.")

    # =========================================================================
    # SEÇÃO 4: FUNDAMENTAÇÃO TEÓRICA DA MODELAGEM
    # =========================================================================
    add_custom_heading1(doc, "4. Seção 4: Fundamentação Teórica da Modelagem")

    add_custom_heading2(doc, "4.1 O Paradigma de Transfer Learning")
    add_body_paragraph(doc,
        "Treinar uma rede neural profunda a partir de pesos inicializados aleatoriamente (from scratch) para um domínio de 38 classes exigiria "
        "milhões de amostras adicionais e semanas de computação apenas para que os primeiros filtros aprendessem primitivas visuais elementares "
        "(arestas, gradientes de cor e texturas de Gabor). Pelo paradigma de Transfer Learning (Aprendizado por Transferência Indutiva), "
        "aproveita-se a representação de características de alto nível aprendida por uma rede treinada sobre o acervo ImageNet-1k. "
        "Na etapa de adaptação, preservaram-se os pesos convolucionais como inicialização informada, substituiu-se o classificador linear final "
        "original (1.000 classes) por uma camada customizada (Linear(1280, 38)) e treinou-se o conjunto ponta a ponta com taxa de aprendizado "
        "controlada (eta = 10^-4), permitindo refinamento fino acelerado.")

    add_custom_heading2(doc, "4.2 Seleção da Arquitetura MobileNetV2 para Edge AI")
    add_body_paragraph(doc,
        "A escolha da MobileNetV2 (Sandler et al., 2018) reflete diretamente o modelo de negócio da AgroSense Analytics: o aplicativo móvel "
        "destina-se a produtores rurais em campo. Redes como a ResNet-50 (~25,6 milhões de parâmetros) ou modelos baseados em Vision Transformers "
        "(ViT, ~86 milhões de parâmetros) impõem latência excessiva, alto consumo de bateria e sobrecarga inviável de memória RAM em smartphones "
        "intermediários. A MobileNetV2 alcança acurácia comparável consumindo apenas ~3,5 milhões de parâmetros, permitindo inferências locais "
        "instantâneas (abaixo de 50 milissegundos por quadro).")

    add_custom_heading2(doc, "4.3 Dedução Matemática: Convoluções Separáveis em Profundidade")
    add_body_paragraph(doc,
        "O diferencial de eficiência da MobileNetV2 decorre da substituição da convolução padrão bidimensional por uma Convolução Separável "
        "em Profundidade (Depthwise Separable Convolution), que decompõe a filtragem espacial da combinação linear de canais:")

    add_body_paragraph(doc,
        "Considere um mapa de ativação de entrada de dimensões D_F x D_F com M canais, e um kernel de tamanho D_K x D_K projetando N canais de saída:")

    add_formula_block(doc, "Custo_conv = D_K · D_K · M · N · D_F · D_F", "Convolução Clássica")
    add_formula_block(doc, "Custo_depthwise = D_K · D_K · M · D_F · D_F", "Etapa Depthwise")
    add_formula_block(doc, "Custo_pointwise = 1 · 1 · M · N · D_F · D_F", "Etapa Pointwise 1x1")
    add_formula_block(doc, "Custo_sep = D_K² · M · D_F² + M · N · D_F²", "Custo Total Separável")

    add_body_paragraph(doc, "Calculando a razão analítica de redução de operações multiplicativas (FLOPs):")
    add_formula_block(doc, "Razão = (D_K² · M · D_F² + M · N · D_F²) / (D_K² · M · N · D_F²) = 1/N + 1/D_K²", "Fator de Redução")

    add_body_paragraph(doc,
        "Para filtros espaciais típicos de 3 x 3 (D_K = 3) e grande número de canais de saída (N >> 9), a razão aproxima-se de 1 / 3² = 1/9 ~ 0,11. "
        "Isso comprova matematicamente que a MobileNetV2 requer entre 8 a 9 vezes menos operações computacionais que uma convolução clássica, "
        "preservando praticamente a mesma capacidade de representação expressiva.")

    add_custom_heading2(doc, "4.4 Blocos Residuais Invertidos e Gargalo Linear (Linear Bottlenecks)")
    add_body_paragraph(doc,
        "A MobileNetV2 introduz o conceito de Inverted Residual Block: recebe um bloco comprimido de baixa dimensionalidade, aplica uma expansão "
        "de canal por um fator t = 6 via convolução 1x1, processa a convolução em profundidade no espaço hiperdimensional, e projeta de volta "
        "para baixa dimensionalidade. A não-linearidade (ReLU6) é deliberadamente removida da última projeção 1x1. A fundamentação teórica "
        "demonstra que aplicar ReLU em espaços de baixa dimensão colapsa irreversivelmente a variedade diferenciável dos dados (manifold destruction), "
        "destruindo informações úteis. O gargalo linear preserva a riqueza espectral da representação, enquanto conexões residuais (skip connections) "
        "conectam diretamente os gargalos de entrada e saída, mitigando o problema do desvanecimento do gradiente.")

    # =========================================================================
    # SEÇÃO 5: RESULTADOS, MÉTRICAS E ANÁLISE CRÍTICA
    # =========================================================================
    add_custom_heading1(doc, "5. Seção 5: Resultados, Métricas e Análise Crítica")

    add_custom_heading2(doc, "5.1 Formalização Matemática das Métricas de Classificação Multiclasse")
    add_body_paragraph(doc,
        "Para avaliar formalmente a capacidade preditiva do modelo sobre as 38 classes, estruturou-se a Matriz de Confusão Multiclasse "
        "C em N^(K x K) (onde K = 38), cujos elementos C_i,j denotam a contagem de amostras pertencentes à classe verdadeira i classificadas "
        "pelo modelo como classe j. A partir da matriz, definem-se formalmente as equações analíticas:")

    add_formula_block(doc, "Acurácia = (Σ TP_i) / N_total = (Σ C_i,i) / (ΣΣ C_i,j)", "Acurácia Global Top-1")
    add_formula_block(doc, "Precisão_i = TP_i / (TP_i + FP_i) = C_i,i / (Σ_k C_k,i)", "Precisão da Classe i")
    add_formula_block(doc, "Revocação_i = TP_i / (TP_i + FN_i) = C_i,i / (Σ_k C_i,k)", "Revocação da Classe i")
    add_formula_block(doc, "F1_i = 2 · (Precisão_i · Revocação_i) / (Precisão_i + Revocação_i)", "F1-Score da Classe i")
    add_formula_block(doc, "Macro F1 = (1 / K) · Σ F1_i", "Macro F1-Score")
    add_formula_block(doc, "Weighted F1 = Σ (N_i / N_total) · F1_i", "Weighted F1-Score")

    add_custom_heading2(doc, "5.2 O Paradoxo da Acurácia em Conjuntos Desbalanceados")
    add_body_paragraph(doc,
        "O Paradoxo da Acurácia ocorre quando um modelo atinge acurácia global alta simplesmente por acertar as classes majoritárias, "
        "enquanto falha sistematicamente nas classes raras. No agronegócio da AgroSense Analytics, deixar de identificar uma doença rara "
        "(Falso Negativo) pode significar a perda total da lavoura do produtor. Portanto, apoiar a validação apenas na Acurácia Global "
        "de 99,41% seria negligente. A confirmação da robustez exige que o Macro F1-Score (que atribui o mesmo peso a classes com 23 e "
        "com 826 amostras) acompanhe a acurácia de ponta.")

    add_custom_heading2(doc, "5.3 Resultados Reais Aferidos no Conjunto de Teste Inédito (N = 8.146)")
    add_body_paragraph(doc,
        "A execução do script de avaliação cega (src/evaluate_metrics.py) consolidou os resultados empíricos no conjunto de teste independente:")

    # Tabela 4: Metricas de Avaliacao
    tbl4_headers = ["Indicador de Desempenho", "Valor Aferido", "Interpretação Analítica e de Negócio para a AgroSense"]
    tbl4_data = [
        ["Acurácia Global (Top-1)", "99,41%", "8.098 acertos exatos em 8.146 folhas de teste completamente inéditas."],
        ["Acurácia Top-3", "99,96%", "Em 99,96% das inferências, o diagnóstico real está contido no top 3 retornado pelo app."],
        ["Acurácia Top-5", "99,98%", "Praticamente nula a ocorrência de omissões completas de diagnóstico patológico."],
        ["Macro F1-Score", "99,19%", "Eficácia balanceada e uniforme comprovada em todas as 38 classes sem favorecer majoritárias."],
        ["Weighted F1-Score", "99,41%", "Desempenho consolidado ponderado pelo suporte de cada cultura agrícola."],
        ["Macro Recall", "99,38%", "Capacidade média de rastreamento das patologias foliares presentes em campo."],
        ["Macro Precision", "99,07%", "Confiabilidade média dos alarmes e laudos fitossanitários gerados."]
    ]
    create_styled_table(doc, tbl4_headers, tbl4_data, [Cm(4.5), Cm(2.5), Cm(9.0)])

    add_custom_heading3(doc, "Auditoria Específica da Classe Minoritária (Potato___healthy)")
    add_body_paragraph(doc,
        "Como comprovação contundente da eficácia da função CrossEntropyLoss com pesos ponderados (calculate_class_weights), "
        "a classe mais rara do acervo (batata sadia) obteve desempenho perfeito no teste cego: suporte de 23 imagens, Revocação (Recall) "
        "de 100,00% (todas as 23 folhas sadias identificadas), Precisão de 100,00% (zero falsos positivos) e F1-Score de 100,00%.")

    add_custom_heading3(doc, "Análise das Discrepâncias Morfológicas e Cuidados com Shortcut Learning")
    add_body_paragraph(doc,
        "A inspeção do classification_report.csv revelou que a classe com menor precisão foi Corn Cercospora leaf spot Gray leaf spot "
        "(P = 83,70%, R = 100,00%, F1 = 91,12%, suporte de 77 amostras). A matriz de confusão demonstrou que as confusões residuais "
        "ocorreram quase exclusivamente com Corn Northern Leaf Blight (R = 89,80%). Na fitopatologia real, a Cercosporiose e a Mancha de "
        "Helminthosporium no milho manifestam lesões necróticas foliares lineares a retangulares morfologicamente idênticas nos estágios iniciais, "
        "explicando a proximidade dos vetores latentes da rede. Por fim, para blindar o modelo contra o fenômeno de Shortcut Learning "
        "(aprendizado de atalhos em bancadas de laboratório), o pipeline implementou fortes perturbações de ColorJitter e rotações estocásticas, "
        "pavimentando a robustez para a Etapa 3 (Consolidação do Modelo).")

    add_custom_heading2(doc, "5.4 Catálogo Sequencial de Figuras de Modelagem e Avaliação")
    add_body_paragraph(doc,
        "Abaixo são apresentadas as três figuras técnicas decorrentes do ciclo de treinamento e avaliação do modelo supervisionado:")

    # Figura 5
    add_custom_heading3(doc, "Curvas de Aprendizado e Convergência da MobileNetV2")
    fig5_path = Path("reports/figures/curvas_aprendizado.png")
    if fig5_path.exists():
        add_figure_with_caption(doc, fig5_path,
            "Figura 5 — Curvas de Convergência de Treinamento e Validação: Função de Perda e Acurácia.", width=Cm(14.5))
    add_body_paragraph(doc,
        "A Figura 5 demonstra a evolução decrescente da perda ponderada e a subida consistente da acurácia global em treino e validação, "
        "culminando com o checkpoint ótimo salvo na Época 5 (Val Loss: 0,0189; Val Acc: 99,42%).")

    # Figura 6
    add_custom_heading3(doc, "Matriz de Confusão Multiclasse Normalizada")
    fig6_path = Path("reports/figures/matriz_confusao.png")
    if fig6_path.exists():
        add_figure_with_caption(doc, fig6_path,
            "Figura 6 — Matriz de Confusão Multiclasse Normalizada por Revocação (Recall).", width=Cm(14.5))
    add_body_paragraph(doc,
        "A Figura 6 exibe o mapa de calor 38x38 normalizado pelo recall de cada classe, comprovando que a diagonal principal se mantém "
        "praticamente intacta em 1,00 para quase a totalidade das 38 patologias avaliadas no teste inédito.")

    # Figura 7
    add_custom_heading3(doc, "Galeria de Inferência Visual e Diagnóstico de Campo")
    fig7_path = Path("reports/figures/galeria_inferencia.png")
    if fig7_path.exists():
        add_figure_with_caption(doc, fig7_path,
            "Figura 7 — Diagnósticos Preditos em Amostras de Teste Inéditas com Confiabilidade.", width=Cm(14.5))
    add_body_paragraph(doc,
        "A Figura 7 ilustra 16 amostras de teste cego selecionadas aleatoriamente, exibindo os diagnósticos reais versus preditos "
        "pela IA da AgroSense Analytics acompanhados da respectiva probabilidade Softmax de confiança.")

    # Salvar documento Word oficial
    output_path = Path("Relatorio_Etapa2_AgroSense.docx")
    doc.save(str(output_path))
    size_mb = os.path.getsize(output_path) / (1024.0 * 1024.0)
    print(f"[SUCESSO] Documento Word gerado em: {output_path.resolve()}")
    print(f"[INFO] Tamanho final do arquivo: {size_mb:.2f} MB")
    return output_path, size_mb


if __name__ == '__main__':
    out_file, sz = build_document()
    if sz < 1.5:
        print(f"[ERRO] Tamanho do arquivo ({sz:.2f} MB) abaixo do limite de 1.5 MB.", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)
