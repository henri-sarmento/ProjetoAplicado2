"""
AgroSense Analytics — Gerador e Exportador de Figuras da Analise Exploratoria (EDA)
Gera os 4 graficos fundamentais em alta resolucao (300 DPI) para o relatorio da Etapa 2:
1. eda_distribuicao_classes.png
2. eda_status_sanitario.png
3. eda_mosaico_patologias.png
4. eda_distribuicao_rgb.png
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image


def resolve_file(candidates):
    for c in candidates:
        p = Path(c).resolve()
        if p.exists():
            return p
    raise FileNotFoundError(f"Nenhum dos caminhos candidatos foi encontrado: {candidates}")


def resolve_base_dir(df, sample_rel):
    candidates = [
        Path('.'), Path('ProjetoAplicado2'), Path('..'),
        Path('plantvillage dataset/color')
    ]
    for cand in candidates:
        if (cand / sample_rel).exists():
            return cand.resolve()
    return Path('.').resolve()


def save_to_output_dirs(fig, filename, dpi=300):
    output_dirs = [
        Path('reports/figures'),
        Path('ProjetoAplicado2/reports/figures')
    ]
    saved_paths = []
    for od in output_dirs:
        od.mkdir(parents=True, exist_ok=True)
        out_path = od / filename
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        size_kb = os.path.getsize(out_path) / 1024.0
        print(f"[OK] Gerado: {out_path.resolve()} ({size_kb:.1f} KB)")
        saved_paths.append((out_path, size_kb))
    return saved_paths


def plot_distribuicao_classes(df):
    """
    Figura 1: Grafico de barras horizontais ordenado com as 38 classes.
    Destaque visual para as classes extremas (majoritaria e minoritaria).
    """
    print("[1/4] Gerando eda_distribuicao_classes.png...")
    counts = df['classe_rotulo'].value_counts(ascending=True)
    total_imgs = len(df)
    
    fig, ax = plt.subplots(figsize=(15, 18))
    
    # Paleta de cores: Cinza/Verde para intermediarias, Laranja-Escuro para maximo, Roxo/Vermelho para minimo
    max_cls = counts.index[-1]
    min_cls = counts.index[0]
    
    colors = []
    for cls in counts.index:
        if cls == max_cls:
            colors.append('#d32f2f')  # Destaque majoritaria (Vermelho escuro)
        elif cls == min_cls:
            colors.append('#7b1fa2')  # Destaque minoritaria (Roxo vibrante)
        elif 'healthy' in cls.lower():
            colors.append('#2e7d32')  # Saudavel (Verde escuro)
        else:
            colors.append('#455a64')  # Patologia (Cinza ardósia)

    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor='black', linewidth=0.6, height=0.75)
    
    # Anotacoes em cada barra
    for bar in bars:
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        pct = (w / total_imgs) * 100
        ax.text(w + 35, y, f"{w:,} ({pct:.1f}%)", va='center', ha='left', fontsize=8.5, fontweight='bold', color='#263238')

    ax.set_xlim(0, max(counts.values) * 1.15)
    ax.set_title('Distribuicao de Frequencia das 38 Classes Fitopatologicas (PlantVillage - RGB)',
                 fontsize=15, fontweight='bold', pad=20, color='#1a237e')
    ax.set_xlabel('Quantidade Absoluta de Imagens (Amostras)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Classe Fitopatologica (Especie + Patologia)', fontsize=12, fontweight='bold', labelpad=10)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    # Legenda explicativa customizada
    legend_elements = [
        patches.Patch(facecolor='#d32f2f', edgecolor='black', label=f'Classe Majoritaria: {max_cls} ({counts[max_cls]:,} fotos)'),
        patches.Patch(facecolor='#7b1fa2', edgecolor='black', label=f'Classe Minoritaria: {min_cls} ({counts[min_cls]:,} fotos)'),
        patches.Patch(facecolor='#2e7d32', edgecolor='black', label='Espécimes Saudaveis (Healthy)'),
        patches.Patch(facecolor='#455a64', edgecolor='black', label='Espécimes Infectados (Doencas)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.95, facecolor='#f5f5f5')

    # Caixa de destaque metodologico (Callout de Desbalanceamento)
    ratio = counts[max_cls] / counts[min_cls]
    callout_text = (
        f"DIAGNOSTICO DE DESBALANCEAMENTO SEVERO:\n"
        f"- Razao Max/Min: {ratio:.1f} : 1 ({counts[max_cls]:,} vs. {counts[min_cls]} amostras)\n"
        f"- Justifica o uso mandatorio de CrossEntropyLoss Ponderada e Split Estratificado!"
    )
    ax.text(0.55, 0.40, callout_text, transform=ax.transAxes,
            fontsize=10.5, fontweight='bold', color='#b71c1c',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#ffebee', edgecolor='#d32f2f', lw=1.5))

    plt.tight_layout()
    save_to_output_dirs(fig, 'eda_distribuicao_classes.png', dpi=300)
    plt.close(fig)


def plot_status_sanitario(df):
    """
    Figura 2: Proporcao agregada de folhas sadias vs infectadas (barras + rosca).
    """
    print("[2/4] Gerando eda_status_sanitario.png...")
    counts = df['is_saudavel'].value_counts()
    total = len(df)
    
    n_doentes = counts.get(False, 0)
    n_saudaveis = counts.get(True, 0)
    pct_doentes = (n_doentes / total) * 100
    pct_saudaveis = (n_saudaveis / total) * 100

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Grafico de Barras (Contagem Absoluta)
    labels = ['Infectadas / Fitopatologias (False)', 'Saudaveis / Higidas (True)']
    vals = [n_doentes, n_saudaveis]
    bar_colors = ['#d32f2f', '#2e7d32']

    bars = axes[0].bar(labels, vals, color=bar_colors, edgecolor='black', width=0.55, alpha=0.9)
    for bar in bars:
        h = bar.get_height()
        pct = (h / total) * 100
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 800, f"{h:,}\n({pct:.1f}%)",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    axes[0].set_ylim(0, max(vals) * 1.18)
    axes[0].set_title('Contagem Absoluta por Condicao Sanitaria', fontsize=13, fontweight='bold', pad=15)
    axes[0].set_ylabel('Total de Imagens', fontsize=11, fontweight='bold')
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)

    # 2. Grafico de Rosca / Donut (Proporcao Relativa)
    pie_colors = ['#e53935', '#43a047']
    explode = (0.06, 0.0)

    wedges, texts, autotexts = axes[1].pie(
        vals,
        labels=labels,
        autopct='%1.2f%%',
        startangle=140,
        colors=pie_colors,
        explode=explode,
        wedgeprops=dict(width=0.45, edgecolor='black', linewidth=1.2),
        pctdistance=0.75
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(12)
        at.set_fontweight('bold')
    for t in texts:
        t.set_fontsize(10.5)
        t.set_fontweight('bold')

    # Texto central no donut
    axes[1].text(0, 0, f"TOTAL\n{total:,}\nfotos", ha='center', va='center', fontsize=12, fontweight='bold', color='#37474f')
    axes[1].set_title('Proporcao Percentual Relativa no Acervo', fontsize=13, fontweight='bold', pad=15)

    plt.suptitle('AgroSense Analytics — Censo Fitossanitario do Acervo PlantVillage (RGB)',
                 fontsize=15, fontweight='bold', y=1.03, color='#1a237e')
    plt.tight_layout()
    save_to_output_dirs(fig, 'eda_status_sanitario.png', dpi=300)
    plt.close(fig)


def plot_mosaico_patologias(df, base_dir):
    """
    Figura 3: Grade 2x4 com amostras reais de folhas e seus sintomas tipicos.
    """
    print("[3/4] Gerando eda_mosaico_patologias.png...")
    curated = [
        ('Tomato___Target_Spot', 'Necrose Concentrica (Mancha Alvo)', False),
        ('Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Clorose e Enrolamento Foliar (TYLCV)', False),
        ('Corn_(maize)___Common_rust_', 'Pustulas Alaranjadas (Ferrugem Comum)', False),
        ('Apple___healthy', 'Limbo Foliar Integro e Higido', True),
        ('Squash___Powdery_mildew', 'Micelio Branco Pulverulento (Oidio)', False),
        ('Potato___Late_blight', 'Queima Foliar Severa (Requeima)', False),
        ('Grape___Black_rot', 'Lesoes Circulares Escuras (Podridao Negra)', False),
        ('Pepper,_bell___healthy', 'Foliolo Integro sem Infeccao', True)
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for idx, (cls_name, sintoma, is_healthy) in enumerate(curated):
        sub = df[df['classe_rotulo'] == cls_name]
        rel_path = sub.iloc[0]['caminho_arquivo']
        img_path = base_dir / rel_path

        with Image.open(img_path) as im:
            img_rgb = im.convert('RGB')
            axes[idx].imshow(img_rgb)

        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

        # Cor de moldura: Verde para saudavel, Vermelho para doenca
        border_color = '#2e7d32' if is_healthy else '#c62828'
        status_tag = "[SAUDAVEL]" if is_healthy else "[INFECTADA]"
        for spine in axes[idx].spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3.5)

        clean_cls = cls_name.replace('___', ' - ').replace('_', ' ')
        title_text = f"{status_tag}\n{clean_cls}\n({sintoma})"
        axes[idx].set_title(title_text, fontsize=9.5, fontweight='bold', color=border_color, pad=8)

    plt.suptitle('AgroSense Analytics — Mosaico Morfologico de Sintomas Fitopatologicos (PlantVillage)',
                 fontsize=15, fontweight='bold', y=1.02, color='#1a237e')
    plt.tight_layout()
    save_to_output_dirs(fig, 'eda_mosaico_patologias.png', dpi=300)
    plt.close(fig)


def plot_distribuicao_rgb(df):
    """
    Figura 4: Densidade e histogramas de intensidades medias dos canais R, G e B.
    """
    print("[4/4] Gerando eda_distribuicao_rgb.png...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    canais = [
        ('media_canal_r', 'Canal Vermelho (Red)', '#d32f2f'),
        ('media_canal_g', 'Canal Verde (Green)', '#2e7d32'),
        ('media_canal_b', 'Canal Azul (Blue)', '#1565c0')
    ]

    saudaveis = df[df['is_saudavel'] == True]
    doentes = df[df['is_saudavel'] == False]

    for idx, (col, titulo, canal_cor) in enumerate(canais):
        v_h = saudaveis[col]
        v_d = doentes[col]
        
        # Plot KDE / Histograma
        sns.kdeplot(v_h, ax=axes[idx], color='#2e7d32', lw=2.5, label=f'Saudaveis (Med: {v_h.mean():.1f})', fill=True, alpha=0.25)
        sns.kdeplot(v_d, ax=axes[idx], color='#d32f2f', lw=2.5, label=f'Infectadas (Med: {v_d.mean():.1f})', fill=True, alpha=0.25)
        
        # Linhas de media pontilhadas
        axes[idx].axvline(v_h.mean(), color='#1b5e20', linestyle='--', lw=1.8)
        axes[idx].axvline(v_d.mean(), color='#b71c1c', linestyle='--', lw=1.8)

        axes[idx].set_title(f'Assinatura Cromatica — {titulo}', fontsize=12, fontweight='bold', color=canal_cor)
        axes[idx].set_xlabel('Intensidade Media de Pixel (0 - 255)', fontsize=10, fontweight='bold')
        axes[idx].set_ylabel('Densidade de Probabilidade', fontsize=10, fontweight='bold')
        axes[idx].legend(loc='upper right', fontsize=9.5, framealpha=0.9)
        axes[idx].grid(True, linestyle=':', alpha=0.5)

    plt.suptitle('AgroSense Analytics — Assinatura Fotometrica Espectral: Folhas Higidas vs. Infectadas',
                 fontsize=14, fontweight='bold', y=1.02, color='#1a237e')
    plt.tight_layout()
    save_to_output_dirs(fig, 'eda_distribuicao_rgb.png', dpi=300)
    plt.close(fig)


def main():
    csv_path = resolve_file([
        'data/metadados_plantvillage.csv',
        'ProjetoAplicado2/data/metadados_plantvillage.csv',
        '../data/metadados_plantvillage.csv'
    ])
    print(f"[INFO] Carregando metadados de: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Linhas carregadas: {len(df):,}, Colunas: {len(df.columns)}")

    sample_rel = df.iloc[0]['caminho_arquivo']
    base_dir = resolve_base_dir(df, sample_rel)
    print(f"[INFO] Diretorio base das imagens: {base_dir}")

    # Gerar as 4 figuras
    plot_distribuicao_classes(df)
    plot_status_sanitario(df)
    plot_mosaico_patologias(df, base_dir)
    plot_distribuicao_rgb(df)

    # Validacao de parada: verificar se todos os 4 arquivos existem e tem tamanho > 50 KB
    required_files = [
        'eda_distribuicao_classes.png',
        'eda_status_sanitario.png',
        'eda_mosaico_patologias.png',
        'eda_distribuicao_rgb.png'
    ]
    check_dirs = [Path('reports/figures'), Path('ProjetoAplicado2/reports/figures')]
    
    all_ok = True
    print("\n" + "="*70)
    print("VERIFICACAO DE CONFORMIDADE DOS ARTEFATOS:")
    print("="*70)
    for cd in check_dirs:
        for rf in required_files:
            target = cd / rf
            if not target.exists():
                print(f"[FALHA] Arquivo ausente: {target}")
                all_ok = False
            else:
                sz = os.path.getsize(target) / 1024.0
                if sz < 50.0:
                    print(f"[FALHA] Tamanho insuficiente ({sz:.1f} KB < 50 KB): {target}")
                    all_ok = False
                else:
                    print(f"[SUCESSO] {target.name} em {cd}: {sz:.1f} KB (> 50 KB, 300 DPI)")
    print("="*70 + "\n")

    if not all_ok:
        sys.exit(1)

    print("[CONCLUIDO] Todos os 4 graficos de EDA foram exportados com nitidez e alta definicao!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
