"""
AgroSense Analytics — Avaliacao de Metricas e Sanidade do Modelo (Etapa 2)
Calcula e persiste o classification_report completo, F1-Score (Macro e Ponderado),
Top-1 Accuracy e analise da classe minoritaria 'Potato healthy'.
"""

import os
import sys
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from PIL import Image


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_file(rel_paths):
    for p in rel_paths:
        path_obj = Path(p).resolve()
        if path_obj.exists():
            return path_obj
    raise FileNotFoundError(f"Nenhum dos caminhos candidatos foi encontrado: {rel_paths}")


class PlantVillageDataset(Dataset):
    def __init__(self, dataframe, transform=None, base_dir=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

        if base_dir is None:
            candidates = [
                Path('.'), Path('..'), Path('../..'),
                Path('plantvillage dataset/color')
            ]
            sample_rel = self.df.iloc[0]['caminho_arquivo']
            for cand in candidates:
                if (cand / sample_rel).exists():
                    self.base_dir = cand.resolve()
                    break
            else:
                self.base_dir = Path('.').resolve()
        else:
            self.base_dir = Path(base_dir).resolve()

        self.image_paths = self.df['caminho_arquivo'].tolist()
        self.labels = self.df['label_idx'].to_numpy()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.base_dir / self.image_paths[idx]
        with Image.open(img_path) as img:
            image = img.convert('RGB')

        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def load_test_split(csv_path, seed=42):
    df = pd.read_csv(csv_path)
    classes = sorted(df['classe_rotulo'].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}
    df['label_idx'] = df['classe_rotulo'].map(class_to_idx)

    # Split identico ao utilizado no treinamento: 70% Treino, 15% Val, 15% Teste
    df_train, df_temp = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df['classe_rotulo']
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=0.50, random_state=seed, stratify=df_temp['classe_rotulo']
    )

    return df_test, classes, class_to_idx, idx_to_class


def build_model(num_classes=38, device='cpu'):
    try:
        model = models.mobilenet_v2(weights=None)
    except Exception:
        model = models.mobilenet_v2(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=in_features, out_features=num_classes)
    return model.to(device)


def evaluate():
    set_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Dispositivo de avaliacao: {device}")
    if device.type == 'cuda':
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    # 1. Resolver caminhos de checkpoint e metadados
    checkpoint_path = resolve_file([
        'models/mobilenetv2_plantvillage.pth',
        'ProjetoAplicado2/models/mobilenetv2_plantvillage.pth',
        '../models/mobilenetv2_plantvillage.pth'
    ])
    csv_path = resolve_file([
        'data/metadados_plantvillage.csv',
        'ProjetoAplicado2/data/metadados_plantvillage.csv',
        '../data/metadados_plantvillage.csv'
    ])

    print(f"[INFO] Checkpoint carregado de: {checkpoint_path}")
    print(f"[INFO] Metadados carregados de: {csv_path}")

    # 2. Carregar checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = ckpt.get('classes')
    if classes is None:
        raise KeyError("O checkpoint nao possui a lista 'classes'.")

    num_classes = len(classes)
    model = build_model(num_classes=num_classes, device=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 3. Preparar conjunto de teste estratificado
    df_test, meta_classes, class_to_idx, idx_to_class = load_test_split(csv_path, seed=42)
    print(f"[INFO] Total de amostras no conjunto de teste inedito: {len(df_test):,}")

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = PlantVillageDataset(df_test, transform=test_transforms)
    batch_size = 64 if device.type == 'cuda' else 32
    pin_memory = (device.type == 'cuda')
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory
    )

    # 4. Inferência e Coleta de Predições
    print("[INFO] Executando inferencia no conjunto de teste...")
    all_preds = []
    all_targets = []
    all_top3_hits = 0
    all_top5_hits = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device, non_blocking=pin_memory)
            targets = targets.to(device, non_blocking=pin_memory)
            outputs = model(images)

            b_sz = targets.size(0)
            total_samples += b_sz

            _, topk = outputs.topk(5, dim=1)
            for i in range(b_sz):
                t = targets[i].item()
                pk = topk[i].tolist()
                if t in pk[:3]:
                    all_top3_hits += 1
                if t in pk[:5]:
                    all_top5_hits += 1

            all_preds.extend(topk[:, 0].cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 5. Cálculo das Métricas Scikit-Learn
    top1_acc = accuracy_score(all_targets, all_preds) * 100.0
    top3_acc = (all_top3_hits / total_samples) * 100.0
    top5_acc = (all_top5_hits / total_samples) * 100.0

    report_dict = classification_report(
        all_targets, all_preds, target_names=classes, digits=4, output_dict=True
    )
    report_text = classification_report(
        all_targets, all_preds, target_names=classes, digits=4
    )

    macro_f1 = report_dict['macro avg']['f1-score'] * 100.0
    macro_precision = report_dict['macro avg']['precision'] * 100.0
    macro_recall = report_dict['macro avg']['recall'] * 100.0

    weighted_f1 = report_dict['weighted avg']['f1-score'] * 100.0
    weighted_precision = report_dict['weighted avg']['precision'] * 100.0
    weighted_recall = report_dict['weighted avg']['recall'] * 100.0

    # Analise da classe minoritaria 'Potato___healthy'
    potato_healthy_key = 'Potato___healthy'
    potato_data = report_dict.get(potato_healthy_key, {})
    potato_precision = potato_data.get('precision', 0.0) * 100.0
    potato_recall = potato_data.get('recall', 0.0) * 100.0
    potato_f1 = potato_data.get('f1-score', 0.0) * 100.0
    potato_support = int(potato_data.get('support', 0))

    # 6. Salvar arquivos de saída em reports/metrics
    output_dirs = [
        Path('reports/metrics'),
        Path('ProjetoAplicado2/reports/metrics')
    ]
    for od in output_dirs:
        od.mkdir(parents=True, exist_ok=True)

    # 6.1 Salvar classification_report.csv
    report_df = pd.DataFrame(report_dict).T
    for od in output_dirs:
        csv_out = od / 'classification_report.csv'
        report_df.to_csv(csv_out, index=True, float_format='%.4f')
        print(f"[OK] Salvo: {csv_out.resolve()}")

    # 6.2 Salvar metricas_resumo.json
    resumo = {
        "top1_accuracy_percent": round(top1_acc, 2),
        "top3_accuracy_percent": round(top3_acc, 2),
        "top5_accuracy_percent": round(top5_acc, 2),
        "macro_f1_percent": round(macro_f1, 2),
        "macro_precision_percent": round(macro_precision, 2),
        "macro_recall_percent": round(macro_recall, 2),
        "weighted_f1_percent": round(weighted_f1, 2),
        "weighted_precision_percent": round(weighted_precision, 2),
        "weighted_recall_percent": round(weighted_recall, 2),
        "minority_class_analysis": {
            "class_name": potato_healthy_key,
            "support": potato_support,
            "accuracy_recall_percent": round(potato_recall, 2),
            "precision_percent": round(potato_precision, 2),
            "f1_score_percent": round(potato_f1, 2)
        },
        "total_test_samples": total_samples,
        "num_classes": num_classes
    }

    for od in output_dirs:
        json_out = od / 'metricas_resumo.json'
        with open(json_out, 'w', encoding='utf-8') as f:
            json.dump(resumo, f, indent=2, ensure_ascii=False)
        print(f"[OK] Salvo: {json_out.resolve()}")

    # 7. Exibição Consolidada no Terminal
    print("\n" + "="*80)
    print("      AGROSENSE ANALYTICS — RELATORIO CONSOLIDADO DE CLASSIFICACAO (TESTE)")
    print("="*80)
    print(report_text)
    print("="*80)
    print("RESUMO EXECUTIVO DE METRICAS:")
    print(f" - Acuracia Global (Top-1): {top1_acc:.2f}%")
    print(f" - Acuracia Top-3:          {top3_acc:.2f}%")
    print(f" - Acuracia Top-5:          {top5_acc:.2f}%")
    print(f" - Macro F1-Score:          {macro_f1:.2f}%")
    print(f" - Weighted F1-Score:       {weighted_f1:.2f}%")
    print(f" - Macro Recall:            {macro_recall:.2f}%")
    print(f" - Macro Precision:         {macro_precision:.2f}%")
    print("-"*80)
    print(f"AUDITORIA DA CLASSE MINORITARIA ('{potato_healthy_key}'):")
    print(f" - Suporte no teste:        {potato_support} amostras")
    print(f" - Acuracia (Recall):       {potato_recall:.2f}%")
    print(f" - Precision:               {potato_precision:.2f}%")
    print(f" - F1-Score:                {potato_f1:.2f}%")
    print("="*80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(evaluate())
