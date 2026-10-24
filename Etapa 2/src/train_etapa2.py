"""
Script de Treinamento e Avaliação do Modelo da Etapa 2
AgroSense Analytics — Detecção de Fitopatologias no PlantVillage com MobileNetV2
"""
import os
import sys
import time
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def set_seed(seed=42):
    """Fixa sementes para reprodutibilidade estrita."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_directories():
    """Garante que todos os diretórios de saída existam no disco."""
    for d in ['models', 'reports/figures', 'notebooks']:
        os.makedirs(d, exist_ok=True)


def get_transforms():
    """Retorna transformações de treino (Data Augmentation) e validação/teste."""
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transforms, val_test_transforms


class PlantVillageDataset(Dataset):
    """Dataset PyTorch de alto desempenho com leitura RGB defensiva e acesso indexado veloz."""
    def __init__(self, dataframe, transform=None, base_dir=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        
        # Resolução flexível do diretório base de imagens
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

        # OTIMIZAÇÃO: pré-conversão para listas/arrays para evitar df.iloc em loop
        self.image_paths = self.df['caminho_arquivo'].tolist()
        self.labels = self.df['label_idx'].to_numpy()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.base_dir / self.image_paths[idx]
        with Image.open(img_path) as img:
            image = img.convert('RGB')  # Conversão estrita tratando canal alfa (RGBA)
        
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def load_and_split_data(csv_path='data/metadados_plantvillage.csv', seed=42):
    """Carrega metadados e realiza split estratificado 70/15/15 sem data leakage."""
    candidates = [
        Path(csv_path),
        Path('../data/metadados_plantvillage.csv'),
        Path('ProjetoAplicado2/data/metadados_plantvillage.csv')
    ]
    resolved_csv = next((p for p in candidates if p.exists()), None)
    if resolved_csv is None:
        raise FileNotFoundError(f"Metadados não encontrados em: {csv_path}")

    df = pd.read_csv(resolved_csv)
    classes = sorted(df['classe_rotulo'].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}
    df['label_idx'] = df['classe_rotulo'].map(class_to_idx)

    # Split 70/15/15 Estratificado
    df_train, df_temp = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df['classe_rotulo']
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=0.50, random_state=seed, stratify=df_temp['classe_rotulo']
    )

    # Asserções de integridade
    assert len(set(df_train.index) & set(df_val.index)) == 0, "Leakage Treino-Validação!"
    assert len(set(df_train.index) & set(df_test.index)) == 0, "Leakage Treino-Teste!"
    assert len(set(df_val.index) & set(df_test.index)) == 0, "Leakage Validação-Teste!"
    assert len(df_train) + len(df_val) + len(df_test) == len(df), "Soma difere do total!"
    assert df_train['classe_rotulo'].nunique() == 38, "Classes faltantes no Treino!"
    assert df_val['classe_rotulo'].nunique() == 38, "Classes faltantes na Validação!"
    assert df_test['classe_rotulo'].nunique() == 38, "Classes faltantes no Teste!"

    return df_train, df_val, df_test, classes, class_to_idx, idx_to_class


def calculate_class_weights(df_train, num_classes=38):
    """Calcula pesos inversos de frequência para compensar o desbalanceamento de 36:1."""
    counts = np.bincount(df_train['label_idx'], minlength=num_classes).astype(np.float32)
    total_samples = len(df_train)
    weights = total_samples / (num_classes * counts)
    # Normalização para que o peso médio seja 1.0
    weights = weights / weights.mean()
    return weights


def build_mobilenetv2(num_classes=38, device='cpu'):
    """Constrói MobileNetV2 com pesos pré-treinados e classificador adaptado."""
    try:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    except Exception:
        model = models.mobilenet_v2(pretrained=True)
    
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=in_features, out_features=num_classes)
    return model.to(device)


def main():
    """Função principal que orquestra o treinamento e a geração de artefatos."""
    set_seed(42)
    ensure_directories()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo selecionado: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"VRAM Total: {vram:.2f} GB")

    # Ingestão e splits
    df_train, df_val, df_test, classes, class_to_idx, idx_to_class = load_and_split_data()
    train_tf, val_test_tf = get_transforms()

    train_ds = PlantVillageDataset(df_train, transform=train_tf)
    val_ds = PlantVillageDataset(df_val, transform=val_test_tf)
    test_ds = PlantVillageDataset(df_test, transform=val_test_tf)

    batch_size = 64 if device.type == 'cuda' else 32
    num_workers = 0
    pin_memory = (device.type == 'cuda')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # Modelo, Perda Ponderada, Otimizador e Scheduler
    model = build_mobilenetv2(num_classes=38, device=device)
    class_weights = calculate_class_weights(df_train, num_classes=38)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)
    
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Loop de Treinamento (5 épocas)
    num_epochs = 5
    checkpoint_path = Path('models/mobilenetv2_plantvillage.pth')
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0

    print(f"\nIniciando treinamento ({num_epochs} épocas) em {device}...")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        
        for images, targets in train_loader:
            images = images.to(device, non_blocking=pin_memory)
            targets = targets.to(device, non_blocking=pin_memory)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * targets.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100.0

        # Validação
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=pin_memory)
                targets = targets.to(device, non_blocking=pin_memory)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * targets.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == targets).sum().item()
                total_val += targets.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100.0
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_val_loss)

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['lr'].append(current_lr)

        is_best = epoch_val_acc > best_val_acc
        if is_best:
            best_val_acc = epoch_val_acc
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'classes': classes,
                'class_to_idx': class_to_idx,
                'idx_to_class': idx_to_class,
                'arch': 'mobilenet_v2',
                'num_classes': 38
            }, checkpoint_path)

        dt = time.time() - t0
        best_tag = " [MELHOR MODELO SALVO]" if is_best else ""
        print(f"Epoca [{epoch:02d}/{num_epochs:02d}] ({dt:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}% | "
              f"LR: {current_lr:.1e}{best_tag}")

    print(f"\nTreinamento finalizado em {(time.time() - start_time)/60:.2f} minutos.")
    print(f"Melhor checkpoint salvo em: {checkpoint_path} (Acurácia: {best_val_acc:.2f}%)")

    # 1. Salvar Curvas de Aprendizado (300 DPI)
    curvas_path = Path('reports/figures/curvas_aprendizado.png')
    curvas_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ep_rng = range(1, num_epochs + 1)
    ax[0].plot(ep_rng, history['train_loss'], 'o-', label='Train Loss', color='#d32f2f', lw=2)
    ax[0].plot(ep_rng, history['val_loss'], 's--', label='Val Loss', color='#1565c0', lw=2)
    ax[0].axvline(best_epoch, color='#2e7d32', linestyle=':', label=f'Melhor ({best_epoch})')
    ax[0].set_title('Convergência da Perda (CrossEntropy Ponderada)', fontweight='bold')
    ax[0].set_xlabel('Época')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(ep_rng, history['train_acc'], 'o-', label='Train Acc', color='#388e3c', lw=2)
    ax[1].plot(ep_rng, history['val_acc'], 's--', label='Val Acc', color='#f57c00', lw=2)
    ax[1].axvline(best_epoch, color='#2e7d32', linestyle=':', label=f'Melhor ({best_epoch})')
    ax[1].set_title('Evolução da Acurácia Global (%)', fontweight='bold')
    ax[1].set_xlabel('Época')
    ax[1].set_ylabel('Acurácia (%)')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    plt.suptitle('AgroSense Analytics — Curvas de Aprendizado MobileNetV2', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(curvas_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Curvas de aprendizado salvas em: {curvas_path}")

    # 2. Avaliação no Teste e Matriz de Confusão (300 DPI)
    print("\nAvaliando melhor modelo no conjunto de teste inédito...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_preds, all_targets, all_probs = [], [], []
    correct_1, correct_3, correct_5, total_test = 0, 0, 0, 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device, non_blocking=pin_memory)
            targets = targets.to(device, non_blocking=pin_memory)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            b_sz = targets.size(0)
            total_test += b_sz
            _, topk = outputs.topk(5, dim=1)
            for i in range(b_sz):
                t = targets[i].item()
                pk = topk[i].tolist()
                if t == pk[0]: correct_1 += 1
                if t in pk[:3]: correct_3 += 1
                if t in pk[:5]: correct_5 += 1

            all_preds.extend(topk[:, 0].cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    print(f"Resultados no Teste ({total_test:,} amostras):")
    print(f"Acurácia Global (Top-1): {(correct_1/total_test)*100:.2f}%")
    print(f"Acurácia Top-3:          {(correct_3/total_test)*100:.2f}%")
    print(f"Acurácia Top-5:          {(correct_5/total_test)*100:.2f}%")

    # Salvar Matriz de Confusão (300 DPI)
    matriz_path = Path('reports/figures/matriz_confusao.png')
    matriz_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(all_targets, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(24, 20))
    sns.heatmap(cm_norm, annot=False, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão Normalizada (Recall por Classe) — MobileNetV2 AgroSense', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Classe Predita', fontsize=14, fontweight='bold')
    plt.ylabel('Classe Verdadeira', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.savefig(matriz_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Matriz de confusão salva em: {matriz_path}")

    # 3. Salvar Galeria de Inferência (300 DPI)
    galeria_path = Path('reports/figures/galeria_inferencia.png')
    galeria_path.parent.mkdir(parents=True, exist_ok=True)
    sample_indices = np.random.choice(len(df_test), 16, replace=False)
    mean_arr = np.array([0.485, 0.456, 0.406])
    std_arr = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(4, 4, figsize=(18, 18))
    axes = axes.flatten()
    for i, s_idx in enumerate(sample_indices):
        img_tensor, target_idx = test_ds[s_idx]
        target_idx = target_idx.item() if hasattr(target_idx, 'item') else int(target_idx)
        pred_idx = all_preds[s_idx]
        conf = all_probs[s_idx][pred_idx] * 100.0
        
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np * std_arr + mean_arr, 0, 1)
        true_label = idx_to_class[target_idx]
        pred_label = idx_to_class[pred_idx]
        is_ok = (target_idx == pred_idx)
        
        axes[i].imshow(img_np)
        axes[i].axis('off')
        color = '#2e7d32' if is_ok else '#c62828'
        status = "[CORRETO]" if is_ok else "[ERRO]"
        axes[i].set_title(f"{status}\nReal: {true_label[:20]}\nPred: {pred_label[:20]} ({conf:.1f}%)",
                          fontsize=9, fontweight='bold', color=color, pad=6)

    plt.suptitle('AgroSense Analytics — Galeria de Diagnóstico de Campo (16 Amostras Inéditas)', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(galeria_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Galeria de inferência salva em: {galeria_path}")

    print("\n[OK] Pipeline concluido e todos os 4 artefatos foram materializados fisicamente com sucesso!")


if __name__ == '__main__':
    main()
