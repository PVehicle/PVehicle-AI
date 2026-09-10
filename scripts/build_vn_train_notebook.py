"""Sinh notebook huan luyen mo hinh nhan dien xe thi truong Viet Nam.

Khac voi notebook Stanford Cars:
  - Du lieu tai len tu may (khong tai tu Kaggle)
  - Xu ly MAT CAN BANG LOP: trong so trong ham mat mat + lay mau can bang
  - It du lieu hon nhieu (~2.000 anh vs 16.185) nen dung fine-tune nhe hon
  - Bao cao do chinh xac THEO TUNG LOP, khong chi tong the

Cach chay:
    python scripts/build_vn_train_notebook.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import PROJECT_ROOT, get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

OUTPUT_FILE = PROJECT_ROOT / "notebooks" / "train_vn_cars_colab.ipynb"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# =====================================================================
md("""
# Huan luyen mo hinh nhan dien xe Viet Nam — PVehicle-AI

Notebook nay chay tren **Google Colab** (Runtime → Change runtime type →
**T4 GPU**).

## Khac gi voi mo hinh Stanford Cars?

| | Stanford Cars | Xe Viet Nam |
| :--- | :--- | :--- |
| So lop | 196 | ~20 |
| So anh | 16.185 | ~2.000 |
| Nguon | Kaggle | Wikimedia Commons |
| Can bang lop | Deu (~42 anh/lop) | **Lech 14.7 lan** |

## Van de lon nhat: mat can bang lop

Toyota Vios co 206 anh, VinFast Fadil chi co 14 anh — lech **14.7 lan**.

Neu khong xu ly, mo hinh se hoc duoc rang "doan Fadil gan nhu luon sai"
nen no **tranh du doan lop do**. Do chinh xac tong the van cao (vi cac lop
lon dung nhieu) nhung Fadil se gan **0%**.

Notebook nay dung ba ky thuat de giam nhe:

1. **Trong so lop** trong ham mat mat — lop it anh duoc nhan trong so cao
2. **Lay mau can bang** — moi epoch cac lop xuat hien deu nhau
3. **Tang cuong du lieu manh** — bu cho viec it anh

> **Luu y ve nhan du lieu:** anh duoc gan nhan TU DONG tu ten file
> Wikimedia, chua qua buoc duyet bang mat. Mot so nhan co the sai (vi du
> Vios sedan lan hatchback deu co ten file giong nhau). Do chinh xac bao
> cao se **cao hon thuc te** vi tap test cung sai nhan giong tap train.
""")

# =====================================================================
md("""
## 1. Kiem tra GPU
""")

code("""
!nvidia-smi
""")

# =====================================================================
md("""
## 2. Ket noi Google Drive

Luu checkpoint de khong mat tien do khi Colab ngat phien.
""")

code("""
from pathlib import Path

from google.colab import drive

drive.mount('/content/drive')

DRIVE_DIR = Path('/content/drive/MyDrive/PVehicle-AI-VN')
CKPT_DIR = DRIVE_DIR / 'checkpoints'
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path('/content/vn_cars')

print('Checkpoint se luu tai:', CKPT_DIR)
""")

# =====================================================================
md("""
## 3. Cai thu vien

Khong ghim phien ban: Colab thay doi phien ban Python theo thoi gian, ghim
cung se that bai. `onnxscript` la bat buoc cho `torch.onnx.export` tu
PyTorch 2.6.
""")

code("""
!pip install -q onnx onnxruntime onnxscript

import torch
import torchvision

print('torch      ', torch.__version__)
print('torchvision', torchvision.__version__)
print('CUDA        ', torch.cuda.is_available())
""")

# =====================================================================
md("""
## 4. Tai du lieu len

Chay o duoi roi chon **hai thu** tu may:

1. File nen `vn_cars.zip` — nen thu muc `data/processed/vn_cars/` tren may
2. File `vn_dataset_config.json` — trong thu muc `data/` cua du an

### Cach tao file nen tren may

```powershell
.\\.venv\\Scripts\\python.exe scripts/pack_vn_dataset.py
```

Script tu thu nho anh ve 448px truoc khi nen — anh goc tu Wikimedia rat
lon trong khi mo hinh chi dung 224px. File nen giam khoang 78%.
""")

code("""
import shutil
import zipfile

from google.colab import files

if DATA_DIR.exists():
    print('Du lieu da co san, bo qua buoc tai len.')
else:
    print('Chon vn_cars.zip va vn_dataset_config.json:')
    uploaded = files.upload()

    zip_name = next(
        (n for n in uploaded if n.endswith('.zip')), None
    )
    if zip_name is None:
        raise ValueError('Khong thay file .zip nao trong so file da chon.')

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_name) as archive:
        archive.extractall(DATA_DIR)

    config_name = next(
        (n for n in uploaded if n.endswith('.json')), None
    )
    if config_name:
        shutil.copy(config_name, DATA_DIR / 'vn_dataset_config.json')

    print('Da giai nen xong.')

!ls {DATA_DIR}
""")

# =====================================================================
md("""
## 5. Doc cau hinh

File `vn_dataset_config.json` chua san so anh tung lop, trong so da tinh,
va thong so ky thuat cua tung dong xe.
""")

code("""
import json

CONFIG_PATH = DATA_DIR / 'vn_dataset_config.json'
config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))

CLASS_NAMES = config['classes']
NUM_CLASSES = len(CLASS_NAMES)
counts = config['counts']

print(f'So lop : {NUM_CLASSES}')
print(f'So anh : {config["total_images"]}')
print()
print(f'{"Lop":<34} {"Anh":>5} {"Trong so":>9}')
print('-' * 50)
for name in sorted(CLASS_NAMES, key=lambda n: -counts[n]):
    weight = config['class_weights'][name]
    print(f'{name:<34} {counts[name]:>5} {weight:>9.2f}')

ratio = max(counts.values()) / min(counts.values())
print()
print(f'Muc mat can bang: {ratio:.1f}x')
""")

# =====================================================================
md("""
## 6. Chuan bi DataLoader

### Tang cuong du lieu manh hon binh thuong

Tap nay chi co ~2.000 anh (so voi 16.185 cua Stanford Cars), va co lop chi
14 anh. Tang cuong manh giup mo hinh khong hoc vet:

| Phep bien doi | Muc dich |
| :--- | :--- |
| `RandomResizedCrop(scale=0.6-1.0)` | Chiu duoc thay doi khoang cach |
| `RandomHorizontalFlip` | Xe chup tu hai phia |
| `ColorJitter` manh hon | Chiu duoc anh sang khac nhau |
| `RandomRotation(10)` | Chiu duoc anh chup hoi nghieng |
| `RandomErasing(p=0.3)` | Chiu duoc xe bi che mot phan |
""")

code("""
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.3),
])

test_transform = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.14)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_dataset = ImageFolder(DATA_DIR / 'train', transform=train_transform)
test_dataset = ImageFolder(DATA_DIR / 'test', transform=test_transform)

# ImageFolder tu doc ten thu muc lam nhan, da dung thu tu chu cai.
IDX_TO_NAME = train_dataset.classes

print(f'Train: {len(train_dataset)} anh')
print(f'Test : {len(test_dataset)} anh')
print(f'So lop: {len(IDX_TO_NAME)}')
assert len(IDX_TO_NAME) == NUM_CLASSES, 'So lop khong khop cau hinh!'
""")

md("""
### Lay mau can bang

`WeightedRandomSampler` cho phep anh cua lop it duoc lay nhieu lan hon
trong moi epoch, de cac lop xuat hien deu nhau.

Khong dung `shuffle=True` khi da co sampler — hai thu nay xung dot.
""")

code("""
# Trong so cho tung ANH: anh thuoc lop it thi trong so cao.
train_targets = [label for _, label in train_dataset.samples]
class_sample_counts = torch.bincount(
    torch.tensor(train_targets), minlength=NUM_CLASSES
).float()

# Nghich dao so luong: lop it anh -> trong so cao.
sample_weights = (1.0 / class_sample_counts)[train_targets]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),
    replacement=True,
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
)

print(f'Train: {len(train_loader)} batch (co lay mau can bang)')
print(f'Test : {len(test_loader)} batch')
print()
print('So anh moi lop trong tap train:')
for idx, name in enumerate(IDX_TO_NAME):
    print(f'  {name:<34} {int(class_sample_counts[idx]):>4}')
""")

# =====================================================================
md("""
## 7. Dung mo hinh

Van dung **EfficientNet-B0** pretrained ImageNet cho dong bo voi mo hinh
Stanford Cars, nhung chi thay lop phan loai cuoi.

Voi tap nho (~2.000 anh), fine-tune toan bo mang de bi qua khop. Tuy nhien
thu nghiem cho thay dong bang phan lon mang lai lam mo hinh hoc kem hon —
xe Viet Nam khac ImageNet kha nhieu. Nen van fine-tune toan bo nhung dung
**learning rate thap hon** va **weight decay cao hon**.
""")

code("""
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        # Dropout cao hon mac dinh (0.2) vi tap du lieu nho.
        nn.Dropout(p=0.4),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


model = build_model().to(DEVICE)

print(f'Thiet bi : {DEVICE}')
print(f'So lop   : {NUM_CLASSES}')
print(f'Tham so  : {sum(p.numel() for p in model.parameters()):,}')
""")

# =====================================================================
md("""
## 8. Ham luu / nap checkpoint
""")

code("""
CKPT_LAST = CKPT_DIR / 'vn_last.pt'
CKPT_BEST = CKPT_DIR / 'vn_best.pt'


def save_checkpoint(path, model, optimizer, scheduler, epoch, history,
                    best_acc):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'history': history,
        'best_acc': best_acc,
        'class_names': IDX_TO_NAME,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler_state'])
    return ckpt['epoch'], ckpt['history'], ckpt['best_acc']
""")

# =====================================================================
md("""
## 9. Ham train va danh gia

Diem khac biet quan trong: **danh gia theo tung lop**, khong chi tong the.

Voi tap mat can bang, do chinh xac tong the co the cao trong khi vai lop
gan nhu khong hoat dong. Chi so **balanced accuracy** (trung binh do chinh
xac cua tung lop) phan anh dung hon.
""")

code("""
import numpy as np
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, total_correct, total_seen = 0.0, 0, 0

    for images, targets in tqdm(loader, desc='  train', leave=False):
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=DEVICE.type == 'cuda'):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * targets.size(0)
        total_correct += (outputs.argmax(1) == targets).sum().item()
        total_seen += targets.size(0)

    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()
def evaluate(model, loader, criterion):
    \"\"\"Danh gia, tra ve ca chi so tong the lan theo tung lop.\"\"\"
    model.eval()
    total_loss, total_seen = 0.0, 0
    correct_per_class = np.zeros(NUM_CLASSES)
    total_per_class = np.zeros(NUM_CLASSES)
    top5_correct = 0

    for images, targets in tqdm(loader, desc='  eval ', leave=False):
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        with autocast('cuda', enabled=DEVICE.type == 'cuda'):
            outputs = model(images)
            loss = criterion(outputs, targets)

        preds = outputs.argmax(1)
        # top-5 chi co y nghia khi so lop > 5.
        k = min(5, NUM_CLASSES)
        _, top_k = outputs.topk(k, dim=1)
        top5_correct += top_k.eq(
            targets.view(-1, 1)
        ).any(dim=1).sum().item()

        for target, pred in zip(targets.cpu().numpy(), preds.cpu().numpy()):
            total_per_class[target] += 1
            if target == pred:
                correct_per_class[target] += 1

        total_loss += loss.item() * targets.size(0)
        total_seen += targets.size(0)

    per_class_acc = np.divide(
        correct_per_class, total_per_class,
        out=np.zeros(NUM_CLASSES), where=total_per_class > 0,
    )

    return {
        'loss': total_loss / total_seen,
        'top1': correct_per_class.sum() / total_seen,
        'top5': top5_correct / total_seen,
        # Trung binh do chinh xac tung lop — khong bi lop lon lan at.
        'balanced': float(per_class_acc[total_per_class > 0].mean()),
        'per_class': per_class_acc,
    }
""")

# =====================================================================
md("""
## 10. Chay huan luyen

**Trong so lop trong ham mat mat:** lop it anh duoc nhan trong so cao hon,
buoc mo hinh chu y thay vi bo qua.

Ket hop voi lay mau can bang o tren, day la hai lop bao ve khac nhau cho
cung mot van de.
""")

code("""
import time

EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3
LABEL_SMOOTHING = 0.1

# Trong so lop cho ham mat mat, doc tu file cau hinh.
weight_tensor = torch.tensor(
    [config['class_weights'][name] for name in IDX_TO_NAME],
    dtype=torch.float32, device=DEVICE,
)

criterion = nn.CrossEntropyLoss(
    weight=weight_tensor, label_smoothing=LABEL_SMOOTHING
)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS
)
scaler = GradScaler('cuda', enabled=DEVICE.type == 'cuda')

start_epoch, best_acc = 0, 0.0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [],
           'val_top1': [], 'val_top5': [], 'val_balanced': []}

if CKPT_LAST.exists():
    start_epoch, history, best_acc = load_checkpoint(
        CKPT_LAST, model, optimizer, scheduler
    )
    print(f'Tiep tuc tu epoch {start_epoch + 1}'
          f', balanced acc tot nhat: {best_acc:.4f}')
else:
    print('Bat dau huan luyen tu dau.')

for epoch in range(start_epoch, EPOCHS):
    started = time.time()
    print(f'\\nEpoch {epoch + 1}/{EPOCHS}')

    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, scaler
    )
    metrics = evaluate(model, test_loader, criterion)
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(metrics['loss'])
    history['val_top1'].append(metrics['top1'])
    history['val_top5'].append(metrics['top5'])
    history['val_balanced'].append(metrics['balanced'])

    print(
        f'  train loss {train_loss:.4f} | acc {train_acc:.4f}'
        f' || val top-1 {metrics["top1"]:.4f}'
        f' | top-5 {metrics["top5"]:.4f}'
        f' | balanced {metrics["balanced"]:.4f}'
        f' | {time.time() - started:.0f}s'
    )

    save_checkpoint(CKPT_LAST, model, optimizer, scheduler, epoch + 1,
                    history, best_acc)

    # Chon mo hinh tot nhat theo BALANCED accuracy, khong phai top-1.
    # Voi tap mat can bang, top-1 cao co the che giau viec vai lop chet.
    if metrics['balanced'] > best_acc:
        best_acc = metrics['balanced']
        save_checkpoint(CKPT_BEST, model, optimizer, scheduler, epoch + 1,
                        history, best_acc)
        print(f'  -> Tot nhat moi (balanced {best_acc:.4f}), da luu.')

print(f'\\nHoan tat. Balanced accuracy tot nhat: {best_acc:.4f}')
""")

# =====================================================================
md("""
## 11. Do chinh xac theo tung lop

**Day la bang quan trong nhat cua notebook.** No cho thay lop nao that su
hoat dong, lop nao khong.

Do chinh xac tong the cao ma vai lop 0% nghia la mo hinh dang bo qua cac
lop do — dung dua con so tong the vao bao cao ma khong kem bang nay.
""")

code("""
best_model = build_model().to(DEVICE)
best_epoch, _, best_balanced = load_checkpoint(CKPT_BEST, best_model)
best_model.eval()

final = evaluate(best_model, test_loader, criterion)

print(f'Checkpoint tot nhat: epoch {best_epoch}')
print(f'  Top-1              : {final["top1"]:.2%}')
print(f'  Top-5              : {final["top5"]:.2%}')
print(f'  Balanced accuracy  : {final["balanced"]:.2%}')
print()
print(f'{"Lop":<34} {"Anh train":>10} {"Do chinh xac":>13}')
print('-' * 60)

rows = sorted(
    zip(IDX_TO_NAME, final['per_class']),
    key=lambda item: item[1],
)
for name, acc in rows:
    n_train = counts.get(name, 0)
    flag = '  <-- kem' if acc < 0.5 else ''
    print(f'{name:<34} {n_train:>10} {acc:>12.1%}{flag}')

weak = [name for name, acc in rows if acc < 0.5]
if weak:
    print()
    print(f'{len(weak)} lop co do chinh xac duoi 50%.')
    print('Nguyen nhan thuong gap: qua it anh, hoac nhan bi sai.')
""")

# =====================================================================
md("""
## 12. Ma tran nham lan

Cho thay mo hinh hay nham lop nao voi lop nao. Neu hai dong xe cung hang
va cung kieu dang bi nham nhieu, do la dieu de hieu.
""")

code("""
import matplotlib.pyplot as plt


@torch.no_grad()
def confusion_matrix(model, loader):
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    model.eval()
    for images, targets in loader:
        outputs = model(images.to(DEVICE))
        preds = outputs.argmax(1).cpu().numpy()
        for target, pred in zip(targets.numpy(), preds):
            matrix[target, pred] += 1
    return matrix


matrix = confusion_matrix(best_model, test_loader)

fig, ax = plt.subplots(figsize=(12, 10))
image = ax.imshow(matrix, cmap='Blues')
ax.set_xticks(range(NUM_CLASSES))
ax.set_yticks(range(NUM_CLASSES))
# Rut gon ten cho de doc: bo nam va kieu dang.
short = [' '.join(n.split()[:2]) for n in IDX_TO_NAME]
ax.set_xticklabels(short, rotation=90, fontsize=8)
ax.set_yticklabels(short, fontsize=8)
ax.set_xlabel('Du doan')
ax.set_ylabel('Thuc te')
ax.set_title('Ma tran nham lan')
plt.colorbar(image)
plt.tight_layout()
plt.savefig(DRIVE_DIR / 'vn_confusion_matrix.png', dpi=150,
            bbox_inches='tight')
plt.show()

# Liet ke cac cap bi nham nhieu nhat.
print('Cac cap bi nham nhieu nhat:')
pairs = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j and matrix[i, j] > 0:
            pairs.append((matrix[i, j], IDX_TO_NAME[i], IDX_TO_NAME[j]))
for count, actual, predicted in sorted(pairs, reverse=True)[:10]:
    print(f'  {count:>3} lan: {actual[:28]:<28} -> {predicted[:28]}')
""")

# =====================================================================
md("""
## 13. Bieu do qua trinh huan luyen
""")

code("""
epochs_ran = range(1, len(history['train_loss']) + 1)
fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))

ax_loss.plot(epochs_ran, history['train_loss'], label='Train')
ax_loss.plot(epochs_ran, history['val_loss'], label='Validation')
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Loss')
ax_loss.set_title('Loss theo epoch')
ax_loss.legend()
ax_loss.grid(alpha=0.3)

ax_acc.plot(epochs_ran, history['val_top1'], label='Top-1')
ax_acc.plot(epochs_ran, history['val_top5'], label='Top-5')
ax_acc.plot(epochs_ran, history['val_balanced'], label='Balanced',
            linewidth=2, linestyle='--')
ax_acc.set_xlabel('Epoch')
ax_acc.set_ylabel('Do chinh xac')
ax_acc.set_title('Do chinh xac theo epoch')
ax_acc.legend()
ax_acc.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(DRIVE_DIR / 'vn_training_curves.png', dpi=150,
            bbox_inches='tight')
plt.show()
""")

# =====================================================================
md("""
## 14. Export sang ONNX

Giong mo hinh Stanford Cars: `dynamo=False` de gop tat ca vao MOT file,
tranh viec trong so bi tach ra file `.onnx.data` rieng.
""")

code("""
ONNX_PATH = DRIVE_DIR / 'vn_car_classifier.onnx'

dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)

torch.onnx.export(
    best_model,
    dummy_input,
    str(ONNX_PATH),
    input_names=['input'],
    output_names=['logits'],
    dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,
)

size_mb = ONNX_PATH.stat().st_size / 1e6
print(f'Da export: {ONNX_PATH} ({size_mb:.1f} MB)')

if size_mb < 5:
    print()
    print('CANH BAO: file qua nho, trong so co the bi tach ra file rieng.')
""")

md("""
### Kiem tra file ONNX
""")

code("""
import onnx
import onnxruntime as ort

onnx.checker.check_model(onnx.load(str(ONNX_PATH)))
print('Cau truc ONNX hop le.')

session = ort.InferenceSession(str(ONNX_PATH),
                               providers=['CPUExecutionProvider'])

sample = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
with torch.no_grad():
    torch_out = best_model(sample.to(DEVICE)).cpu().numpy()
onnx_out = session.run(None, {'input': sample.numpy()})[0]

max_diff = np.abs(torch_out - onnx_out).max()
print(f'Sai lech lon nhat: {max_diff:.6f}')

if max_diff < 1e-3:
    print('ONNX khop voi PyTorch.')
elif max_diff < 1e-2:
    print('CANH BAO: sai lech hoi lon nhung chap nhan duoc.')
else:
    raise ValueError(f'Sai lech {max_diff:.4f} qua lon!')
""")

# =====================================================================
md("""
## 15. Luu nhan va thong so xe

Ngoai danh sach nhan, luu luon thong so ky thuat cua tung dong xe de app
tra cuu duoc ngay.
""")

code("""
LABELS_PATH = DRIVE_DIR / 'vn_class_names.json'
SPECS_PATH = DRIVE_DIR / 'vn_car_specs.json'

LABELS_PATH.write_text(
    json.dumps(IDX_TO_NAME, ensure_ascii=False, indent=2), encoding='utf-8'
)
SPECS_PATH.write_text(
    json.dumps(config['specs'], ensure_ascii=False, indent=2),
    encoding='utf-8',
)

print(f'Da luu {len(IDX_TO_NAME)} nhan vao {LABELS_PATH.name}')
print(f'Da luu thong so {len(config["specs"])} dong xe')
print()
print('3 nhan dau:', IDX_TO_NAME[:3])
""")

# =====================================================================
md("""
## 16. Tai ket qua ve may

Chep ba file vao thu muc `models/` cua du an:

| File | Dat vao |
| :--- | :--- |
| `vn_car_classifier.onnx` | `models/vn_car_classifier.onnx` |
| `vn_class_names.json` | `models/vn_class_names.json` |
| `vn_car_specs.json` | `models/vn_car_specs.json` |
""")

code("""
files.download(str(ONNX_PATH))
files.download(str(LABELS_PATH))
files.download(str(SPECS_PATH))
""")

md("""
## Buoc tiep theo

1. Chep ba file tren vao `models/`
2. Cap nhat app de chay **ca hai mo hinh**: Stanford Cars (196 lop, xe
   quoc te) va xe Viet Nam (~20 lop), so sanh do tin cay de chon ket qua

## Doc ket qua the nao?

| Chi so | Y nghia |
| :--- | :--- |
| **Top-1** | Ti le doan dung ngay lua chon dau tien |
| **Top-5** | Ti le xe dung nam trong 5 goi y |
| **Balanced** | Trung binh do chinh xac TUNG LOP |

**Balanced accuracy la chi so dang tin nhat** voi tap mat can bang. Neu
top-1 cao ma balanced thap, nghia la mo hinh dang bo qua cac lop it anh.

> Nhac lai: anh chua qua buoc duyet bang mat, nen mot so nhan co the sai.
> Con so bao cao se cao hon thuc te.
""")


def build_notebook() -> dict:
    cells = []
    for cell_type, source in CELLS:
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def main() -> int:
    setup_logging()

    notebook = build_notebook()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    n_code = sum(1 for kind, _ in CELLS if kind == "code")
    n_md = sum(1 for kind, _ in CELLS if kind == "markdown")
    logger.info(
        "Da tao %s (%d o code, %d o markdown)",
        OUTPUT_FILE.name, n_code, n_md,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
