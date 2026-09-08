"""Sinh notebook huan luyen tang phan loai (chay tren Google Colab).

Viet notebook bang script thay vi sua truc tiep file .ipynb, vi JSON cua
notebook rat kho doc va kho review trong git diff.

Cach chay:
    python scripts/build_train_notebook.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import PROJECT_ROOT, get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

OUTPUT_FILE = PROJECT_ROOT / "notebooks" / "train_classifier_colab.ipynb"

# Moi phan tu: ("markdown" | "code", noi dung)
CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# =====================================================================
# 1. Gioi thieu
# =====================================================================
md("""
# Huan luyen tang phan loai xe (196 lop) — PVehicle-AI

Notebook nay chay tren **Google Colab**
(Runtime → Change runtime type → **T4 GPU**).

## Kien truc 2 tang

| Tang | Mo hinh | Huan luyen o dau |
| :--- | :--- | :--- |
| 1. Phat hien xe | YOLOv8 pretrained COCO | Khong can train |
| 2. Phan loai dong xe | EfficientNet-B0 (196 lop) | **Notebook nay** |

Stanford Cars khong co bounding box theo tung dong xe, nen tang phat hien
dung thang model pretrained. Notebook nay chi lo tang thu hai: doc anh, crop
theo bounding box co san trong file `.mat`, roi train phan loai.

## Quy trinh

1. Ket noi Google Drive (luu checkpoint — Colab free ngat sau ~4-6h)
2. Tai dataset tu Kaggle
3. Crop anh theo bounding box
4. Train co resume tu checkpoint
5. Danh gia top-1 / top-5, ve bieu do
6. Export ONNX va tai ve may
""")

# =====================================================================
# 2. Kiem tra GPU
# =====================================================================
md("""
## 1. Kiem tra GPU

Neu lenh duoi bao loi, vao **Runtime → Change runtime type → T4 GPU**.
Train tren CPU se mat nhieu gio thay vi vai chuc phut.
""")

code("""
!nvidia-smi
""")

# =====================================================================
# 3. Mount Drive
# =====================================================================
md("""
## 2. Ket noi Google Drive

**Bat buoc.** Colab free ngat phien lam viec sau khoang 4-6 gio va xoa sach
o dia tam. Neu khong luu checkpoint ra Drive thi moi lan ngat la mat trang
toan bo qua trinh train.
""")

code("""
from pathlib import Path

from google.colab import drive

drive.mount('/content/drive')

# Thu muc luu checkpoint va model. Ton tai lau dai qua nhieu phien Colab.
DRIVE_DIR = Path('/content/drive/MyDrive/PVehicle-AI')
CKPT_DIR = DRIVE_DIR / 'checkpoints'
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Kich thuoc dau vao cua YOLOv8, phai khop src/cv/detector.py.
YOLO_INPUT_SIZE_EXPORT = 640

# Thu muc du lieu, nam tren o dia tam cua Colab (nhanh hon Drive rat nhieu).
DATA_DIR = Path('/content/data')
CROP_DIR = Path('/content/cropped')

print('Checkpoint se luu tai:', CKPT_DIR)
""")

# =====================================================================
# 4. Cai thu vien
# =====================================================================
md("""
## 3. Cai thu vien

Colab da co sang `torch`, `torchvision`, `scipy`. Chi can bo sung cac goi
dung cho export ONNX va thanh tien trinh.

> **Khong ghim phien ban o day.** Colab thay doi phien ban Python theo thoi
> gian; ghim cung se that bai khi ban `onnxruntime` do khong con phat hanh
> cho Python moi. Vi du: tren Python 3.13, `onnxruntime==1.19.2` khong ton
> tai (thap nhat la 1.20.0).
>
> Viec khac phien ban giua Colab va may local khong gay van de, vi mo hinh
> duoc export voi `opset_version=17` — dinh dang on dinh ma ca hai ban deu
> doc duoc.
""")

code("""
!pip install -q onnx onnxruntime kaggle

import scipy
import torch
import torchvision

print('torch      ', torch.__version__)
print('torchvision', torchvision.__version__)
print('scipy      ', scipy.__version__)
print('CUDA        ', torch.cuda.is_available())
""")

# =====================================================================
# 5. Kaggle API
# =====================================================================
md("""
## 4. Cau hinh Kaggle API

Dataset goc tren trang Stanford **da bi go bo**, nen phai tai qua Kaggle.

### Cach lay API token

1. Tao tai khoan tai <https://www.kaggle.com>
2. Vao **Settings** (bam vao anh dai dien) → muc **API**
3. Bam **Create New Token**
4. Copy chuoi token (dang `KGAT_...`) va dan vao o nhap ben duoi

> **Luu y bao mat:** o nhap dung `getpass` nen token **khong hien ra man
> hinh va khong bi luu vao file notebook**. Tuyet doi khong go thang token
> vao o code — notebook luu ca ket qua chay, token se lo ra khi chia se.
>
> Kaggle chi cho xem token dung mot lan. Neu lo, vao lai Settings → API →
> **Create New Token** de sinh token moi (token cu tu het hieu luc).

Notebook ho tro ca hai co che xac thuc cua Kaggle:

| Co che | Cach dung |
| :--- | :--- |
| **Token moi** (khuyen nghi) | Dan chuoi `KGAT_...` vao o nhap |
| **`kaggle.json` cu** | Neu da co san file, notebook tu dung, khong hoi |
""")

code("""
import os
from getpass import getpass
from pathlib import Path

KAGGLE_JSON = Path.home() / '.kaggle' / 'kaggle.json'
ACCESS_TOKEN_FILE = Path.home() / '.kaggle' / 'access_token'


def setup_kaggle_auth():
    \"\"\"Cau hinh xac thuc Kaggle, uu tien thong tin da co san.

    Thu tu kiem tra khop voi thu tu ma thu vien kaggle tu tim:
    bien moi truong -> file access_token -> file kaggle.json cu.
    \"\"\"
    if os.environ.get('KAGGLE_API_TOKEN'):
        return 'bien moi truong KAGGLE_API_TOKEN'
    if ACCESS_TOKEN_FILE.exists():
        return f'file {ACCESS_TOKEN_FILE}'
    if KAGGLE_JSON.exists():
        return f'file {KAGGLE_JSON} (co che cu)'

    # getpass: token khong hien ra man hinh, khong luu vao notebook.
    token = getpass('Dan Kaggle API token (dang KGAT_...): ').strip()
    if not token:
        raise ValueError('Chua nhap token. Hay chay lai o nay.')

    os.environ['KAGGLE_API_TOKEN'] = token
    return 'token vua nhap'


source = setup_kaggle_auth()
print('Xac thuc bang:', source)
""")

# =====================================================================
# 6. Tai dataset
# =====================================================================
md("""
## 5. Tai dataset Stanford Cars

Dung ban `rickyyyyyyy/torchvision-stanford-cars` (~2 GB) vi ban nay giu
nguyen cac file `.mat` goc, nghia la co:

- **Bounding box** cho tung anh (can de crop cho tang phan loai)
- **Ten lop chuan** khop 100% voi `data/car_specs.csv` cua du an

Cac ban Kaggle khac (vi du `jutrera`) doi ten thu muc thanh dang
`2012 Tesla Model S` (nam o dau) va **mat bounding box**, nen khong dung.

Buoc nay mat khoang 2-4 phut.
""")

code("""
KAGGLE_DATASET = 'rickyyyyyyy/torchvision-stanford-cars'
STANFORD_DIR = DATA_DIR / 'stanford_cars'

if STANFORD_DIR.exists():
    print('Dataset da co san, bo qua buoc tai.')
else:
    # Import sau khi da dat KAGGLE_API_TOKEN: thu vien kaggle doc bien
    # moi truong ngay luc import, dat sau se khong an.
    import kaggle

    try:
        kaggle.api.authenticate()
    except Exception as exc:
        raise RuntimeError(
            'Xac thuc Kaggle that bai. Kiem tra lai token con hieu luc '
            'khong, hoac tao token moi tai Settings -> API.'
        ) from exc

    print(f'Dang tai {KAGGLE_DATASET} (~2 GB), doi vai phut...')
    kaggle.api.dataset_download_files(
        KAGGLE_DATASET, path=str(DATA_DIR), unzip=True
    )
    print('Tai xong.')

!ls -la {STANFORD_DIR}
""")

# =====================================================================
# 7. Doc annotation
# =====================================================================
md("""
## 6. Doc nhan va bounding box

Cau truc thu muc sau khi giai nen:

```text
data/stanford_cars/
├── cars_train/                        # 8.144 anh
├── cars_test/                         # 8.041 anh
├── cars_test_annos_withlabels.mat     # nhan + bbox cua tap test
└── devkit/
    ├── cars_train_annos.mat           # nhan + bbox cua tap train
    └── cars_meta.mat                  # ten cua 196 lop
```

`torchvision.datasets.StanfordCars` chi tra ve anh va nhan, **bo qua bounding
box**. Vi kien truc 2 tang can crop dung vung xe, ta doc thang file `.mat`
bang `scipy` de lay them toa do.
""")

code("""
import scipy.io as sio

DEVKIT_DIR = STANFORD_DIR / 'devkit'


def load_annotations(mat_path, images_dir):
    \"\"\"Doc file .mat, tra ve danh sach (duong_dan_anh, nhan, bbox).

    Nhan trong file .mat danh so tu 1, doi ve 0 cho khop voi PyTorch.
    \"\"\"
    raw = sio.loadmat(str(mat_path), squeeze_me=True)['annotations']
    samples = []
    for item in raw:
        samples.append({
            'path': images_dir / str(item['fname']),
            'label': int(item['class']) - 1,
            'bbox': (
                int(item['bbox_x1']), int(item['bbox_y1']),
                int(item['bbox_x2']), int(item['bbox_y2']),
            ),
        })
    return samples


CLASS_NAMES = sio.loadmat(
    str(DEVKIT_DIR / 'cars_meta.mat'), squeeze_me=True
)['class_names'].tolist()

train_samples = load_annotations(
    DEVKIT_DIR / 'cars_train_annos.mat', STANFORD_DIR / 'cars_train'
)
test_samples = load_annotations(
    STANFORD_DIR / 'cars_test_annos_withlabels.mat',
    STANFORD_DIR / 'cars_test',
)

print(f'So lop        : {len(CLASS_NAMES)}')
print(f'Anh train     : {len(train_samples)}')
print(f'Anh test      : {len(test_samples)}')
print(f'Lop dau tien  : {CLASS_NAMES[0]}')
print(f'Lop cuoi cung : {CLASS_NAMES[-1]}')
print(f'Vi du mau     : {train_samples[0]}')

assert len(CLASS_NAMES) == 196, 'Phai co dung 196 lop'
""")

# =====================================================================
# 8. Crop anh
# =====================================================================
md("""
## 7. Crop anh theo bounding box

Tang phan loai chi can vung chua xe. Crop truoc mot lan roi luu ra dia se
nhanh hon nhieu so voi crop lai o moi epoch.

Bounding box duoc noi rong them **8%** moi chieu: giu lai mot phan boi canh
giup mo hinh nhan dang tot hon, va tranh cat mat vien xe khi hop hoi lech.

Buoc nay mat khoang 3-5 phut cho 16.185 anh.
""")

code("""
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from tqdm.auto import tqdm

BBOX_PADDING = 0.08  # noi rong hop 8% moi chieu


def crop_one(sample, out_dir):
    \"\"\"Crop mot anh theo bbox (co noi rong) va luu vao thu muc theo lop.\"\"\"
    out_path = out_dir / str(sample['label']) / sample['path'].name
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(sample['path']) as img:
        img = img.convert('RGB')
        width, height = img.size
        x1, y1, x2, y2 = sample['bbox']

        pad_x = int((x2 - x1) * BBOX_PADDING)
        pad_y = int((y2 - y1) * BBOX_PADDING)
        # Ep toa do nam trong khung anh, tranh loi khi hop cham vien.
        box = (
            max(x1 - pad_x, 0), max(y1 - pad_y, 0),
            min(x2 + pad_x, width), min(y2 + pad_y, height),
        )
        img.crop(box).save(out_path, quality=95)


def crop_split(samples, out_dir, desc):
    out_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(tqdm(
            pool.map(lambda s: crop_one(s, out_dir), samples),
            total=len(samples), desc=desc,
        ))


crop_split(train_samples, CROP_DIR / 'train', 'Crop train')
crop_split(test_samples, CROP_DIR / 'test', 'Crop test')

n_train = len(list((CROP_DIR / 'train').rglob('*.jpg')))
n_test = len(list((CROP_DIR / 'test').rglob('*.jpg')))
print(f'Da crop: {n_train} anh train, {n_test} anh test')
""")

md("""
### Xem thu vai anh sau khi crop

Buoc kiem tra bang mat: neu anh bi cat sai hoac lech nhan thi phat hien ngay
o day, truoc khi ton hang tieng train.
""")

code("""
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, sample in zip(axes.ravel(), train_samples[:8]):
    cropped = CROP_DIR / 'train' / str(sample['label']) / sample['path'].name
    ax.imshow(Image.open(cropped))
    ax.set_title(CLASS_NAMES[sample['label']], fontsize=9)
    ax.axis('off')
plt.tight_layout()
plt.show()
""")

# =====================================================================
# 9. DataLoader
# =====================================================================
md("""
## 8. Chuan bi DataLoader

**Tang cuong du lieu (chi ap dung cho tap train):**

| Phep bien doi | Muc dich |
| :--- | :--- |
| `RandomResizedCrop` | Mo hinh chiu duoc thay doi khoang cach chup |
| `RandomHorizontalFlip` | Xe chup tu trai va tu phai deu nhan duoc |
| `ColorJitter` | Chiu duoc dieu kien anh sang khac nhau |
| `RandomErasing` | Chiu duoc truong hop xe bi che mot phan |

Tap test **khong** tang cuong, chi resize va chuan hoa, de ket qua danh gia
phan anh dung nang luc that.
""")

code("""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 2

# Gia tri chuan hoa cua ImageNet (mo hinh pretrained duoc train voi bo nay).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.25),
])

test_transform = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.14)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_dataset = ImageFolder(CROP_DIR / 'train', transform=train_transform)
test_dataset = ImageFolder(CROP_DIR / 'test', transform=test_transform)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
)

# ImageFolder sap xep thu muc theo thu tu chuoi ('0','1','10','100'...),
# KHONG phai thu tu so. Phai anh xa lai de nhan khop voi CLASS_NAMES.
FOLDER_TO_CLASS = {
    idx: int(folder_name)
    for folder_name, idx in train_dataset.class_to_idx.items()
}
IDX_TO_NAME = [
    CLASS_NAMES[FOLDER_TO_CLASS[i]] for i in range(len(train_dataset.classes))
]

print(f'Train : {len(train_dataset)} anh, {len(train_loader)} batch')
print(f'Test  : {len(test_dataset)} anh, {len(test_loader)} batch')
print(f'Kiem tra anh xa nhan — lop 0 la: {IDX_TO_NAME[0]}')
""")

# =====================================================================
# 10. Model
# =====================================================================
md("""
## 9. Dung mo hinh

Dung **EfficientNet-B0** pretrained ImageNet, thay lop cuoi thanh 196 lop.

Ly do chon B0: nhe (~5,3 trieu tham so), suy luan nhanh tren CPU sau khi
export ONNX, ma do chinh xac van du tot cho bai toan nay.
""")

code("""
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = len(train_dataset.classes)


def build_model():
    \"\"\"EfficientNet-B0 pretrained, thay lop phan loai cuoi.\"\"\"
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return model


model = build_model().to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f'Thiet bi   : {DEVICE}')
print(f'So lop     : {NUM_CLASSES}')
print(f'Tham so    : {n_params:,}')
""")

# =====================================================================
# 11. Checkpoint
# =====================================================================
md("""
## 10. Ham luu / nap checkpoint

**Phan quan trong nhat cua notebook nay.** Colab free ngat phien sau ~4-6h.
Moi epoch deu luu lai trang thai day du (model, optimizer, scheduler, so
epoch, lich su) ra Drive.

Khi bi ngat: chay lai notebook tu dau, o train se tu dong doc checkpoint va
**hoc tiep tu epoch dang do**, khong mat gi.
""")

code("""
CKPT_LAST = CKPT_DIR / 'last.pt'
CKPT_BEST = CKPT_DIR / 'best.pt'


def save_checkpoint(path, model, optimizer, scheduler, epoch, history,
                    best_acc):
    \"\"\"Luu toan bo trang thai de co the hoc tiep sau khi Colab ngat.\"\"\"
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
    \"\"\"Nap checkpoint. Tra ve (epoch_da_xong, history, best_acc).\"\"\"
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler_state'])
    return ckpt['epoch'], ckpt['history'], ckpt['best_acc']
""")

# =====================================================================
# 12. Train / eval
# =====================================================================
md("""
## 11. Ham train va danh gia

Dung **mixed precision** (`torch.amp`) de tang toc khoang 2 lan tren GPU T4
va giam bo nho, nho do giu duoc batch size 64.

Danh gia bao cao ca **top-1** va **top-5**. Voi bai toan 196 lop chi tiet
(nhieu dong xe rat giong nhau), top-5 phan anh do huu ich thuc te tot hon.
""")

code("""
from torch.amp import GradScaler, autocast


def accuracy_topk(outputs, targets, topk=(1, 5)):
    \"\"\"Dem so du doan dung o top-1 va top-5.\"\"\"
    max_k = max(topk)
    _, pred = outputs.topk(max_k, dim=1)
    correct = pred.eq(targets.view(-1, 1))
    return [correct[:, :k].any(dim=1).sum().item() for k in topk]


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
    model.eval()
    total_loss, top1, top5, total_seen = 0.0, 0, 0, 0

    for images, targets in tqdm(loader, desc='  eval ', leave=False):
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        with autocast('cuda', enabled=DEVICE.type == 'cuda'):
            outputs = model(images)
            loss = criterion(outputs, targets)

        c1, c5 = accuracy_topk(outputs, targets)
        total_loss += loss.item() * targets.size(0)
        top1 += c1
        top5 += c5
        total_seen += targets.size(0)

    return total_loss / total_seen, top1 / total_seen, top5 / total_seen
""")

# =====================================================================
# 13. Vong lap train
# =====================================================================
md("""
## 12. Chay huan luyen

Cau hinh: **20 epoch**, tren GPU T4 mat khoang 45-60 phut.

Neu Colab ngat giua chung, chi can chay lai o nay — no se tu tim
`last.pt` tren Drive va hoc tiep tu dung cho dang do.
""")

code("""
import time

EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS
)
scaler = GradScaler('cuda', enabled=DEVICE.type == 'cuda')

# Tiep tuc tu checkpoint neu co.
start_epoch, best_acc = 0, 0.0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [],
           'val_top1': [], 'val_top5': []}

if CKPT_LAST.exists():
    start_epoch, history, best_acc = load_checkpoint(
        CKPT_LAST, model, optimizer, scheduler
    )
    print(f'Tiep tuc tu epoch {start_epoch + 1}'
          f', top-1 tot nhat: {best_acc:.4f}')
else:
    print('Bat dau huan luyen tu dau.')

for epoch in range(start_epoch, EPOCHS):
    started = time.time()
    print(f'\\nEpoch {epoch + 1}/{EPOCHS}')

    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, scaler
    )
    val_loss, val_top1, val_top5 = evaluate(model, test_loader, criterion)
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_top1'].append(val_top1)
    history['val_top5'].append(val_top5)

    print(
        f'  train loss {train_loss:.4f} | acc {train_acc:.4f}'
        f' || val loss {val_loss:.4f}'
        f' | top-1 {val_top1:.4f} | top-5 {val_top5:.4f}'
        f' | {time.time() - started:.0f}s'
    )

    # Luu sau MOI epoch de khong mat tien do khi Colab ngat.
    save_checkpoint(CKPT_LAST, model, optimizer, scheduler, epoch + 1,
                    history, best_acc)
    if val_top1 > best_acc:
        best_acc = val_top1
        save_checkpoint(CKPT_BEST, model, optimizer, scheduler, epoch + 1,
                        history, best_acc)
        print(f'  -> Mo hinh tot nhat moi (top-1 {best_acc:.4f}), da luu.')

print(f'\\nHoan tat. Top-1 tot nhat: {best_acc:.4f}')
""")

# =====================================================================
# 14. Bieu do
# =====================================================================
md("""
## 13. Bieu do qua trinh huan luyen

Hai bieu do nay dua thang vao chuong ket qua cua bao cao.
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

ax_acc.plot(epochs_ran, history['train_acc'], label='Train top-1')
ax_acc.plot(epochs_ran, history['val_top1'], label='Val top-1')
ax_acc.plot(epochs_ran, history['val_top5'], label='Val top-5')
ax_acc.set_xlabel('Epoch')
ax_acc.set_ylabel('Do chinh xac')
ax_acc.set_title('Do chinh xac theo epoch')
ax_acc.legend()
ax_acc.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(DRIVE_DIR / 'training_curves.png', dpi=150,
            bbox_inches='tight')
plt.show()

print(f'Top-1 cuoi cung : {history["val_top1"][-1]:.4f}')
print(f'Top-5 cuoi cung : {history["val_top5"][-1]:.4f}')
""")

# =====================================================================
# 15. Export ONNX
# =====================================================================
md("""
## 14. Export sang ONNX

Nap lai **mo hinh tot nhat** (khong phai epoch cuoi) roi export.

Dung **kich thuoc co dinh 224x224**, khong dung `dynamic_axes` cho chieu
rong/cao. Ly do: anh dau vao luon duoc resize ve 224 truoc khi suy luan,
nen truc dong chi lam ONNX Runtime kho toi uu hon ma khong duoc loi ich gi.
Rieng **batch** thi de dong de sau nay xu ly nhieu xe trong mot anh.
""")

code("""
ONNX_PATH = DRIVE_DIR / 'car_classifier.onnx'

best_model = build_model().to(DEVICE)
best_epoch, _, best_top1 = load_checkpoint(CKPT_BEST, best_model)
best_model.eval()
print(f'Nap checkpoint tot nhat: epoch {best_epoch}, top-1 {best_top1:.4f}')

dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)

torch.onnx.export(
    best_model,
    dummy_input,
    str(ONNX_PATH),
    input_names=['input'],
    output_names=['logits'],
    # Chi de batch dong; chieu rong/cao co dinh 224.
    dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
    opset_version=17,
    do_constant_folding=True,
)

size_mb = ONNX_PATH.stat().st_size / 1e6
print(f'Da export: {ONNX_PATH} ({size_mb:.1f} MB)')
""")

md("""
### Kiem tra file ONNX

Doi chieu ket qua giua PyTorch va ONNX Runtime. Neu sai lech qua nguong,
file ONNX bi loi va khong dung duoc — phai phat hien ngay tai day chu khong
phai luc chay app.
""")

code("""
import numpy as np
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

# Nguong 1e-3: du chat de bat loi export, du rong de bo qua sai so lam tron
# giua cac phien ban onnxruntime khac nhau.
if max_diff < 1e-3:
    print('ONNX cho ket qua trung khop voi PyTorch.')
elif max_diff < 1e-2:
    print('CANH BAO: sai lech hoi lon nhung van chap nhan duoc.')
    print('Thu do lai do chinh xac bang notebook nay neu ket qua bat thuong.')
else:
    raise ValueError(
        f'Sai lech {max_diff:.4f} qua lon — file ONNX co the bi loi. '
        'Kiem tra lai buoc export truoc khi dung mo hinh nay.'
    )

# Kiem tra them: hai ban phai cho cung thu tu xep hang cac lop.
torch_top5 = torch_out[0].argsort()[::-1][:5]
onnx_top5 = onnx_out[0].argsort()[::-1][:5]
if (torch_top5 == onnx_top5).all():
    print('Thu tu top-5 giong nhau giua hai ban.')
else:
    print('CANH BAO: thu tu top-5 khac nhau giua PyTorch va ONNX!')
""")

# =====================================================================
# 16. Luu nhan
# =====================================================================
md("""
## 15. Export mo hinh phat hien YOLOv8

Tang 1 dung YOLOv8 pretrained COCO, **khong can huan luyen**. Nhung van
phai export sang ONNX de chay duoc tren may local ma khong can cai
`ultralytics` va `torch` (~2,5 GB).

Export ngay tai day vi Colab da co san torch. Chi mat vai chuc giay.
""")

code("""
!pip install -q ultralytics

from ultralytics import YOLO

YOLO_ONNX = DRIVE_DIR / 'yolov8n.onnx'

if YOLO_ONNX.exists():
    print('Da co san yolov8n.onnx, bo qua.')
else:
    detector = YOLO('yolov8n.pt')  # tu tai trong so pretrained COCO
    exported = detector.export(
        format='onnx',
        imgsz=YOLO_INPUT_SIZE_EXPORT,
        opset=17,
        simplify=True,
        # Khong nhung NMS: code local tu xu ly (src/cv/detector.py).
        nms=False,
    )
    Path(exported).replace(YOLO_ONNX)
    print('Da export:', YOLO_ONNX)

print(f'Kich thuoc: {YOLO_ONNX.stat().st_size / 1e6:.1f} MB')
""")

md("""
### Kiem tra dinh dang dau ra cua YOLOv8

Code suy luan local phu thuoc vao dang `[1, 84, 8400]`. Kiem tra ngay de
tranh phat hien sai lech luc chay app.
""")

code("""
yolo_session = ort.InferenceSession(
    str(YOLO_ONNX), providers=['CPUExecutionProvider']
)
yolo_input = yolo_session.get_inputs()[0]
yolo_output = yolo_session.get_outputs()[0]

print('Dau vao :', yolo_input.name, yolo_input.shape)
print('Dau ra  :', yolo_output.name, yolo_output.shape)

expected = [1, 84, 8400]
assert list(yolo_output.shape) == expected, (
    f'Dang dau ra {yolo_output.shape} khac ky vong {expected}. '
    'Phai cap nhat lai src/cv/detector.py cho khop.'
)
print('Dinh dang dau ra dung nhu code local mong doi.')
""")

md("""
## 16. Luu danh sach nhan

File nay cho biet chi so dau ra cua mo hinh ung voi ten xe nao. Bat buoc
phai co, neu khong app se khong dich duoc ket qua suy luan.
""")

code("""
import json

LABELS_PATH = DRIVE_DIR / 'class_names.json'
LABELS_PATH.write_text(
    json.dumps(IDX_TO_NAME, ensure_ascii=False, indent=2), encoding='utf-8'
)

print(f'Da luu {len(IDX_TO_NAME)} nhan vao {LABELS_PATH}')
print('3 nhan dau:', IDX_TO_NAME[:3])
""")

# =====================================================================
# 17. Tai ve
# =====================================================================
md("""
## 17. Tai ket qua ve may

Chep **ca ba file** duoi vao thu muc `models/` cua du an:

| File | Dat vao | Vai tro |
| :--- | :--- | :--- |
| `yolov8n.onnx` | `models/yolov8n.onnx` | Tang 1: phat hien xe |
| `car_classifier.onnx` | `models/car_classifier.onnx` | Tang 2: phan loai |
| `class_names.json` | `models/class_names.json` | 196 nhan |

Ca ba deu da nam san trong Google Drive (`MyDrive/PVehicle-AI/`), co the
tai truc tiep tu do neu trinh duyet chan lenh tai o duoi.
""")

code("""
from google.colab import files

for path in (YOLO_ONNX, ONNX_PATH, LABELS_PATH):
    files.download(str(path))
""")

md("""
## Buoc tiep theo

Sau khi da chep ba file vao `models/`, chay ung dung tren may local:

```powershell
.\\.venv\\Scripts\\streamlit.exe run app.py
```

Chi tiet:
- Quy trinh huan luyen: `docs/training_classifier.md`
- Luong suy luan local: `docs/inference_pipeline.md`
""")


def build_notebook() -> dict:
    """Dung cau truc JSON cua notebook tu danh sach CELLS."""
    cells = []
    for cell_type, source in CELLS:
        # Notebook luu source dang danh sach dong, moi dong giu ky tu \n.
        lines = source.splitlines(keepends=True)
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": lines,
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
