"""
AgroSense Analytics - Extração de Metadados e Atributos de Imagem
Projeto Aplicado II - Ciência de Dados
Autor: Equipe AgroSense Analytics
"""

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image


def process_single_image(args):
    """
    Processa uma única imagem e extrai seus atributos físicos e estatísticas RGB.
    """
    img_path_str, relative_path, class_folder = args
    try:
        stat = os.stat(img_path_str)
        size_kb = round(stat.st_size / 1024.0, 2)
        
        parts = class_folder.split('___')
        especie = parts[0]
        condicao = parts[1]
        is_healthy = (condicao.lower() == 'healthy')
        
        filename = os.path.basename(img_path_str)
        
        with Image.open(img_path_str) as im:
            largura, altura = im.size
            formato = im.format if im.format else 'JPEG'
            espaco_cores = im.mode
            qtd_canais = len(im.getbands())
            
            # Garantir RGB para cálculo estatístico
            if im.mode != 'RGB':
                im_rgb = im.convert('RGB')
            else:
                im_rgb = im
            
            arr = np.asarray(im_rgb, dtype=np.float32)
            
            # Médias e desvios por canal (R, G, B)
            medias = arr.mean(axis=(0, 1))
            desvios = arr.std(axis=(0, 1))
            
            media_r = round(float(medias[0]), 2)
            media_g = round(float(medias[1]), 2)
            media_b = round(float(medias[2]), 2)
            desvio_r = round(float(desvios[0]), 2)
            desvio_g = round(float(desvios[1]), 2)
            desvio_b = round(float(desvios[2]), 2)
            
        return {
            'caminho_arquivo': relative_path.replace('\\', '/'),
            'nome_arquivo': filename,
            'especie_cultura': especie,
            'condicao_sanitaria': condicao,
            'classe_rotulo': class_folder,
            'is_saudavel': is_healthy,
            'largura_px': largura,
            'altura_px': altura,
            'total_pixels': largura * altura,
            'qtd_canais': qtd_canais,
            'espaco_cores': espaco_cores,
            'tamanho_arquivo_kb': size_kb,
            'formato_arquivo': formato,
            'media_canal_r': media_r,
            'media_canal_g': media_g,
            'media_canal_b': media_b,
            'desvio_canal_r': desvio_r,
            'desvio_canal_g': desvio_g,
            'desvio_canal_b': desvio_b
        }
    except Exception as e:
        print(f"Erro ao processar {img_path_str}: {e}", file=sys.stderr)
        return None


def run_extraction(dataset_dir: str, output_csv: str):
    """
    Executa a extração concorrente de todas as imagens da pasta informada.
    """
    start_time = time.time()
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {dataset_dir}")
        
    print(f"Iniciando escaneamento do diretório: {dataset_path.resolve()}")
    
    tasks = []
    classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    print(f"Classes encontradas: {len(classes)}")
    
    for cls in classes:
        cls_dir = dataset_path / cls
        for img_file in cls_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                # Caminho relativo ao projeto
                rel_path = str(img_file.relative_to(dataset_path.parent.parent))
                tasks.append((str(img_file.resolve()), rel_path, cls))
                
    total_imgs = len(tasks)
    print(f"Total de imagens identificadas: {total_imgs}")
    print(f"Iniciando extração paralela com {os.cpu_count()} workers...")
    
    records = []
    batch_size = 5000
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i, res in enumerate(executor.map(process_single_image, tasks, chunksize=100)):
            if res is not None:
                records.append(res)
            if (i + 1) % batch_size == 0 or (i + 1) == total_imgs:
                pct = ((i + 1) / total_imgs) * 100
                elapsed = time.time() - start_time
                fps = (i + 1) / (elapsed + 1e-6)
                print(f"Progresso: {i+1}/{total_imgs} ({pct:.1f}%) - {fps:.1f} imgs/s - Decorrido: {elapsed:.1f}s")
                
    df = pd.DataFrame(records)
    
    # Conversões explícitas de tipagem
    df['caminho_arquivo'] = df['caminho_arquivo'].astype(str)
    df['nome_arquivo'] = df['nome_arquivo'].astype(str)
    df['especie_cultura'] = df['especie_cultura'].astype(str)
    df['condicao_sanitaria'] = df['condicao_sanitaria'].astype(str)
    df['classe_rotulo'] = df['classe_rotulo'].astype(str)
    df['is_saudavel'] = df['is_saudavel'].astype(bool)
    df['largura_px'] = df['largura_px'].astype('int64')
    df['altura_px'] = df['altura_px'].astype('int64')
    df['total_pixels'] = df['total_pixels'].astype('int64')
    df['qtd_canais'] = df['qtd_canais'].astype('int64')
    df['espaco_cores'] = df['espaco_cores'].astype(str)
    df['tamanho_arquivo_kb'] = df['tamanho_arquivo_kb'].astype('float64')
    df['formato_arquivo'] = df['formato_arquivo'].astype(str)
    df['media_canal_r'] = df['media_canal_r'].astype('float64')
    df['media_canal_g'] = df['media_canal_g'].astype('float64')
    df['media_canal_b'] = df['media_canal_b'].astype('float64')
    df['desvio_canal_r'] = df['desvio_canal_r'].astype('float64')
    df['desvio_canal_g'] = df['desvio_canal_g'].astype('float64')
    df['desvio_canal_b'] = df['desvio_canal_b'].astype('float64')
    
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Salvando DataFrame em CSV: {output_path.resolve()}...")
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    # Salvar também versão comprimida gzip
    csv_gz = output_path.with_suffix('.csv.gz')
    df.to_csv(csv_gz, index=False, compression='gzip', encoding='utf-8')
    
    total_time = time.time() - start_time
    print(f"Concluído com sucesso! {len(df)} imagens processadas em {total_time:.2f} segundos.")
    print(f"Dimensões do DataFrame final: {df.shape}")
    return df


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent.parent
    dataset_color_dir = base_dir / "plantvillage dataset" / "color"
    output_file = base_dir / "ProjetoAplicado2" / "data" / "metadados_plantvillage.csv"
    run_extraction(str(dataset_color_dir), str(output_file))
