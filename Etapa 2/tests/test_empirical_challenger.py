"""
Harness de Testes Empíricos e Desafio Adversarial — Challenger 1
AgroSense Analytics — Etapa 2
"""
import sys
import os
import io

# Garantir compatibilidade com console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent

def main():
    print("=" * 80)
    print("INICIANDO HARNESS ADVERSARIAL — EMPIRICAL CHALLENGER 1")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 1. VALIDAÇÃO DOS METADADOS
    # -------------------------------------------------------------------------
    csv_path = REPO_ROOT / "data" / "metadados_plantvillage.csv"
    if not csv_path.exists():
        print(f"[ERRO FATAL] Arquivo de metadados não encontrado: {csv_path}")
        return 1
        
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    print(f"[OBSERVAÇÃO 1] Total de amostras carregadas: {total_samples:,}")
    assert total_samples == 54305, f"Esperado 54.305, encontrado {total_samples}"
    
    classes = sorted(df['classe_rotulo'].unique())
    num_classes = len(classes)
    print(f"[OBSERVAÇÃO 1.1] Total de classes fitopatológicas: {num_classes}")
    assert num_classes == 38, f"Esperado 38 classes, encontrado {num_classes}"

    # -------------------------------------------------------------------------
    # 2. PARTICIONAMENTO ESTRATIFICADO EM DUPLO ESTÁGIO (70 / 15 / 15)
    # -------------------------------------------------------------------------
    print("\n--- TESTE 1: MATEMÁTICA DO PARTICIONAMENTO 70/15/15 ---")
    df_train, df_temp = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df['classe_rotulo']
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=0.50, random_state=42, stratify=df_temp['classe_rotulo']
    )
    
    n_train = len(df_train)
    n_val = len(df_val)
    n_test = len(df_test)
    sum_splits = n_train + n_val + n_test
    
    print(f"Treino:     {n_train:,} ({n_train / total_samples * 100:.4f}%)")
    print(f"Validação:  {n_val:,} ({n_val / total_samples * 100:.4f}%)")
    print(f"Teste:      {n_test:,} ({n_test / total_samples * 100:.4f}%)")
    print(f"Soma Total: {sum_splits:,} (Total Original: {total_samples:,})")
    
    t1_exact_counts = (n_train == 38013 and n_val == 8146 and n_test == 8146)
    t1_sum_exact = (sum_splits == 54305)
    print(f"-> Contagens exatas esperadas (38.013 / 8.146 / 8.146)? {t1_exact_counts}")
    print(f"-> Soma fecha perfeitamente em 54.305? {t1_sum_exact}")
    
    assert t1_exact_counts, f"Contagens divergiram: train={n_train}, val={n_val}, test={n_test}"
    assert t1_sum_exact, f"Soma divergiu: {sum_splits} != 54305"

    # -------------------------------------------------------------------------
    # 3. TESTE DE VAZAMENTO DE DADOS (DATA LEAKAGE)
    # -------------------------------------------------------------------------
    print("\n--- TESTE 2: AUDITORIA ADVERSARIAL DE VAZAMENTO (DATA LEAKAGE) ---")
    idx_train = set(df_train.index)
    idx_val = set(df_val.index)
    idx_test = set(df_test.index)
    
    leak_idx_tv = idx_train.intersection(idx_val)
    leak_idx_tt = idx_train.intersection(idx_test)
    leak_idx_vt = idx_val.intersection(idx_test)
    
    print(f"Interseção de índices Treino ∩ Validação: {len(leak_idx_tv)}")
    print(f"Interseção de índices Treino ∩ Teste:     {len(leak_idx_tt)}")
    print(f"Interseção de índices Validação ∩ Teste:  {len(leak_idx_vt)}")
    
    paths_train = set(df_train['caminho_arquivo'])
    paths_val = set(df_val['caminho_arquivo'])
    paths_test = set(df_test['caminho_arquivo'])
    
    leak_path_tv = paths_train.intersection(paths_val)
    leak_path_tt = paths_train.intersection(paths_test)
    leak_path_vt = paths_val.intersection(paths_test)
    
    print(f"Interseção de caminhos Treino ∩ Validação: {len(leak_path_tv)}")
    print(f"Interseção de caminhos Treino ∩ Teste:     {len(leak_path_tt)}")
    print(f"Interseção de caminhos Validação ∩ Teste:  {len(leak_path_vt)}")
    
    union_paths = paths_train | paths_val | paths_test
    print(f"União dos caminhos cobre todo o acervo? {len(union_paths)} == {total_samples}")
    
    assert len(leak_idx_tv) == 0 and len(leak_idx_tt) == 0 and len(leak_idx_vt) == 0, "Leakage de índice detectado!"
    assert len(leak_path_tv) == 0 and len(leak_path_tt) == 0 and len(leak_path_vt) == 0, "Leakage de caminho detectado!"
    assert len(union_paths) == total_samples, "União das partições não fecha o total!"

    # -------------------------------------------------------------------------
    # 4. TESTE DE REPRESENTAÇÃO DAS CLASSES E CASO LIMITE (Potato___healthy)
    # -------------------------------------------------------------------------
    print("\n--- TESTE 3: REPRESENTAÇÃO DAS 38 CLASSES E CASO POTATO___HEALTHY ---")
    train_counts = df_train['classe_rotulo'].value_counts()
    val_counts = df_val['classe_rotulo'].value_counts()
    test_counts = df_test['classe_rotulo'].value_counts()
    
    print(f"Classes presentes no Treino:    {len(train_counts)} / 38")
    print(f"Classes presentes na Validação: {len(val_counts)} / 38")
    print(f"Classes presentes no Teste:     {len(test_counts)} / 38")
    
    assert len(train_counts) == 38, "Treino possui classes ausentes!"
    assert len(val_counts) == 38, "Validação possui classes ausentes!"
    assert len(test_counts) == 38, "Teste possui classes ausentes!"
    
    zero_train = [c for c in classes if train_counts.get(c, 0) == 0]
    zero_val = [c for c in classes if val_counts.get(c, 0) == 0]
    zero_test = [c for c in classes if test_counts.get(c, 0) == 0]
    assert not zero_train and not zero_val and not zero_test, "Zero-dropout detectado!"
    
    menor_classe = "Potato___healthy"
    tot_potato = (df['classe_rotulo'] == menor_classe).sum()
    n_pot_train = train_counts.get(menor_classe, 0)
    n_pot_val = val_counts.get(menor_classe, 0)
    n_pot_test = test_counts.get(menor_classe, 0)
    
    print(f"\nDetalhes da menor classe ({menor_classe}):")
    print(f"  Total no dataset: {tot_potato}")
    print(f"  No Treino:        {n_pot_train} ({n_pot_train / tot_potato * 100:.1f}%)")
    print(f"  Na Validação:     {n_pot_val} ({n_pot_val / tot_potato * 100:.1f}%)")
    print(f"  No Teste:         {n_pot_test} ({n_pot_test / tot_potato * 100:.1f}%)")
    print(f"  Soma das partes:  {n_pot_train + n_pot_val + n_pot_test}")
    
    assert tot_potato == 152, f"Esperado 152 para Potato___healthy, obtido {tot_potato}"
    assert n_pot_train == 106, f"Esperado 106 no treino, obtido {n_pot_train}"
    assert n_pot_val == 23, f"Esperado 23 na validação, obtido {n_pot_val}"
    assert n_pot_test == 23, f"Esperado 23 no teste, obtido {n_pot_test}"
    assert n_pot_train + n_pot_val + n_pot_test == 152, "Soma de Potato___healthy divergiu!"

    # -------------------------------------------------------------------------
    # 5. TESTE DE IMAGEM RGBA / ROBUSTEZ CROMÁTICA
    # -------------------------------------------------------------------------
    print("\n--- TESTE 4: TRATAMENTO ROBUSTO DE IMAGEM RGBA (4 CANAIS) ---")
    rgba_meta = df[df['espaco_cores'] == 'RGBA']
    print(f"Imagens registradas com espaco_cores == 'RGBA' no CSV: {len(rgba_meta)}")
    
    if len(rgba_meta) > 0:
        rgba_row = rgba_meta.iloc[0]
        print(f"  Caminho da imagem RGBA: {rgba_row['caminho_arquivo']}")
        print(f"  Classe: {rgba_row['classe_rotulo']}")
        print(f"  Qtd Canais no CSV: {rgba_row['qtd_canais']}")
    else:
        print("  Buscando por imagens com 4 canais em todo o CSV...")
        four_channels = df[df['qtd_canais'] == 4]
        print(f"  Imagens com qtd_canais == 4: {len(four_channels)}")
        if len(four_channels) > 0:
            print(f"  Exemplo: {four_channels.iloc[0]['caminho_arquivo']}")
            
    # Teste empírico de crash com simulação RGBA
    print("\n[Simulação de Falha sem .convert('RGB')]")
    synth_rgba = Image.new("RGBA", (256, 256), color=(100, 150, 80, 255))
    
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    tensor_raw = to_tensor(synth_rgba)
    print(f"Tensor cru sem conversão: shape={tensor_raw.shape}, dtype={tensor_raw.dtype}")
    
    crash_reproduced = False
    error_msg = ""
    try:
        norm(tensor_raw)
    except Exception as e:
        crash_reproduced = True
        error_msg = str(e)
        print(f"✓ Crash reproduzido com sucesso! Erro: {type(e).__name__}: {e}")
        
    assert crash_reproduced, "A normalização ImageNet deveria falhar em tensor de 4 canais!"
    
    # Teste com defesa .convert('RGB')
    print("\n[Defesa com .convert('RGB')]")
    clean_rgb = synth_rgba.convert('RGB')
    assert clean_rgb.mode == 'RGB'
    assert len(clean_rgb.getbands()) == 3
    tensor_clean = to_tensor(clean_rgb)
    assert tensor_clean.shape == torch.Size([3, 256, 256])
    normalized_clean = norm(tensor_clean)
    assert normalized_clean.shape == torch.Size([3, 256, 256])
    print(f"✓ Normalização bem-sucedida com shape [3, 256, 256], min={normalized_clean.min():.2f}, max={normalized_clean.max():.2f}")

    # -------------------------------------------------------------------------
    # 6. TESTE DE INTEGRIDADE DO NOTEBOOK JUPYTER (JSON PARSING)
    # -------------------------------------------------------------------------
    print("\n--- TESTE 5: INTEGRIDADE DO NOTEBOOK JUPYTER ---")
    nb_path = REPO_ROOT / "notebooks" / "Etapa_2_Treinamento_Modelo_e_Metricas.ipynb"
    print(f"Lendo notebook: {nb_path}")
    
    nb_valid_json = True
    nb_error = None
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)
        print("✓ Notebook carregado com sucesso via json.load!")
        print(f"  Total de células: {len(nb_data['cells'])}")
    except Exception as e:
        nb_valid_json = False
        nb_error = str(e)
        print(f"[FALHA CRÍTICA DE JSON NO NOTEBOOK]: {type(e).__name__}: {e}")

    # Verificar escape sequences no arquivo cru
    with open(nb_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
        
    invalid_escapes = []
    for line_num, line in enumerate(raw_lines, 1):
        for col_num in range(len(line) - 1):
            if line[col_num] == '\\' and line[col_num + 1] not in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']:
                invalid_escapes.append((line_num, col_num + 1, line[col_num:col_num + 2], line.strip()[:80]))
                
    if invalid_escapes:
        print(f"\n[ATENÇÃO] Encontradas {len(invalid_escapes)} sequências de escape inválidas no JSON do notebook:")
        for line_num, col_num, esc, snippet in invalid_escapes[:10]:
            print(f"  Linha {line_num}, Col {col_num}: escape '{esc}' -> {snippet}")

    print("\n" + "=" * 80)
    print("RESUMO DO HARNESS ADVERSARIAL:")
    print(f"  1. Particionamento 70/15/15:        PASS (38.013 / 8.146 / 8.146 = 54.305)")
    print(f"  2. Prova de Zero Data Leakage:      PASS (Disjuntos)")
    print(f"  3. Representação de Classes:        PASS (Potato___healthy 106 / 23 / 23 = 152)")
    print(f"  4. Tratamento de Imagens RGBA:      PASS (Defesa .convert('RGB') validada)")
    print(f"  5. Integridade do Notebook JSON:    {'PASS' if nb_valid_json else 'FAIL (' + str(nb_error) + ')'}")
    print("=" * 80)
    
    return 0 if nb_valid_json else 2

if __name__ == '__main__':
    sys.exit(main())
