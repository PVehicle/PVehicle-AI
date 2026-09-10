# PVehicle-AI — Đặc tả Frontend

Tài liệu này dành cho việc xây dựng giao diện web cho hệ thống nhận diện và
tư vấn ô tô PVehicle-AI. Backend đã hoàn thiện và đang chạy — frontend chỉ
cần gọi REST API.

> **Lưu ý về tên gọi:** dự án dùng **Angular** (phiên bản hiện đại, v22),
> **không phải AngularJS** (1.x). AngularJS đã ngừng hỗ trợ từ 01/2022 và
> không còn nhận bản vá bảo mật.

---

## 1. Bối cảnh

### Hệ thống làm gì

Người dùng tải lên ảnh ô tô → hệ thống nhận diện dòng xe → hiển thị thông
số kỹ thuật và gợi ý các xe tương tự. Ngoài ra có chức năng tư vấn xe theo
nhu cầu (ngân sách, số chỗ, kiểu dáng).

### Kiến trúc backend

```text
Ảnh → YOLOv8n (phát hiện xe) → crop → 2 mô hình phân loại song song
                                        ├── 196 dòng xe quốc tế
                                        └── 20 dòng xe Việt Nam
                                              ↓
                                    Thông số + xe tương tự
```

Toàn bộ chạy bằng ONNX Runtime trên CPU. Một ảnh mất khoảng **135 ms**.

### Độ chính xác hiện tại

| Mô hình | Top-1 | Top-5 |
| :--- | ---: | ---: |
| Xe quốc tế (196 lớp) | 83.0% | 95.3% |
| Xe Việt Nam (20 lớp) | 75.2% | 92.0% |

---

## 2. Công nghệ

| Hạng mục | Lựa chọn | Phiên bản |
| :--- | :--- | :--- |
| Framework | Angular (standalone components) | 22.1 |
| Ngôn ngữ | TypeScript (strict mode) | 7.0 |
| Quản lý trạng thái | Angular Signals + `@ngrx/signals` | 22.0 |
| HTTP | `HttpClient` với `provideHttpClient(withFetch())` | — |
| Giao diện | Angular Material + Tailwind CSS | 22.1 / 4.3 |
| Kiểm thử | Vitest + Angular Testing Library | 5.0 |
| Build | Angular CLI (esbuild) | 22.1 |

### Nguyên tắc bắt buộc

**Standalone components.** Không dùng `NgModule` — Angular 15+ đã chuyển
sang standalone và từ v19 đây là mặc định.

**Signals thay vì RxJS cho trạng thái UI.** Chỉ dùng RxJS cho luồng bất
đồng bộ (HTTP, debounce). Trạng thái đồng bộ dùng `signal()`,
`computed()`, `resource()`.

**Điều khiển luồng mới.** Dùng `@if`, `@for`, `@switch` thay cho
`*ngIf`, `*ngFor`, `*ngSwitch`.

**Zoneless change detection.** Angular 22 hỗ trợ `provideZonelessChangeDetection()`
— dùng nó, hiệu năng tốt hơn và không cần `zone.js`.

**`input()` / `output()` functions** thay cho decorator `@Input()` /
`@Output()`.

---

## 3. Đặc tả API

File OpenAPI đầy đủ: [`openapi.json`](./openapi.json)

Sinh TypeScript types tự động:

```bash
npx openapi-typescript docs/frontend/openapi.json -o src/app/core/api-types.ts
```

### Base URL

```text
http://localhost:8000
```

Cấu hình qua `environment.ts`, không viết cứng trong service.

### Xác thực

Nếu backend bật API key, mọi request phải kèm header:

```http
X-API-Key: <key>
```

Hai endpoint `/health` và `/ready` **luôn mở**.

> Khi phát triển, backend thường chạy không cần key. Frontend phải hỗ trợ
> cả hai trường hợp — dùng HTTP interceptor, chỉ thêm header khi có key.

### 3.1. Nhận diện xe

```http
POST /api/v1/recognize?top_k=5&include_similar=true&similar_count=3
Content-Type: multipart/form-data
```

| Tham số | Kiểu | Mặc định | Giới hạn |
| :--- | :--- | ---: | :--- |
| `file` | File (form-data) | — | JPEG/PNG/BMP/GIF/WEBP, ≤10 MB |
| `top_k` | query int | 5 | 1-20 |
| `include_similar` | query bool | false | — |
| `similar_count` | query int | 3 | 1-10 |

**Response:**

```json
{
  "vehicle_count": 1,
  "likely_not_a_car": false,
  "processing_ms": 134.82,
  "vehicles": [
    {
      "box": {
        "x1": 12, "y1": 71, "x2": 940, "y2": 606,
        "confidence": 0.6234,
        "coco_class": "car"
      },
      "source": "international",
      "is_confident": true,
      "predictions": [
        {
          "model_index": 95,
          "class_name": "Honda Odyssey Minivan 2012",
          "confidence": 0.8549
        }
      ],
      "specs": {
        "class_id": 185,
        "class_name": "Honda Odyssey Minivan 2012",
        "brand": "Honda",
        "model": "Odyssey",
        "body_style": "Minivan",
        "year": 2012,
        "seats": 7,
        "segment": "economy",
        "price_million_vnd": 709.4,
        "fuel_l_per_100km": 10.0
      },
      "similar_cars": []
    }
  ]
}
```

#### Những điểm dễ hiểu nhầm

**`model_index` ≠ `class_id`.** Hai hệ đánh số khác nhau:
- `model_index` (0-195): chỉ số đầu ra của mô hình
- `class_id` (1-196): số thứ tự trong danh mục Stanford Cars

**Luôn dùng `class_name` để đối chiếu**, đừng dùng số.

**`box` có thể là `null`.** Khi YOLOv8 không tìm thấy xe nào, hệ thống phân
loại toàn bộ ảnh. Giao diện phải xử lý trường hợp này (không vẽ khung).

**`is_confident: false`** nghĩa là xác suất dưới ngưỡng 0.15 — hiển thị
cảnh báo, đừng trình bày như kết luận chắc chắn.

**`likely_not_a_car: true`** nghĩa là nhiều khả năng ảnh không chứa ô tô
(vừa không phát hiện được xe, vừa phân loại không chắc chắn).

**Tọa độ `box` theo ảnh gốc**, không phải ảnh đã resize. Khi vẽ overlay
phải quy đổi theo tỷ lệ hiển thị.

### 3.2. Danh mục xe

```http
GET /api/v1/cars?brand=Toyota&body_style=SUV&limit=20&offset=0
```

Bộ lọc **không phân biệt hoa thường**. Response có `total`, `limit`,
`offset`, `cars[]`.

```http
GET /api/v1/cars/filters
```

Trả về các giá trị hợp lệ để dựng form lọc:

```json
{
  "body_styles": ["Convertible", "Coupe", "Hatchback", "Minivan",
                  "Pickup", "SUV", "Sedan", "Van", "Wagon"],
  "segments": ["economy", "exotic", "luxury"],
  "price_min": 240.0,
  "price_max": 19026.9,
  "seats_min": 2,
  "seats_max": 8
}
```

> **Gọi endpoint này để dựng form**, đừng viết cứng danh sách — bảng thông
> số có thể thay đổi.

```http
GET /api/v1/cars/{class_name}
GET /api/v1/cars/{class_name}/similar?top_n=5
```

`class_name` chứa dấu cách → phải `encodeURIComponent()`.

### 3.3. Tư vấn theo nhu cầu

```http
POST /api/v1/recommend
Content-Type: application/json

{
  "max_price": 800,
  "min_price": 400,
  "min_seats": 7,
  "body_styles": ["SUV", "Minivan"],
  "max_fuel": 9.0,
  "top_n": 5
}
```

Mọi trường đều tùy chọn. Các điều kiện là **ràng buộc cứng** — xe không
thỏa mãn bị loại hẳn, không phải chỉ bị trừ điểm.

**Danh sách rỗng (`count: 0`) không phải lỗi** — nghĩa là không có xe nào
thỏa mãn. Giao diện nên gợi ý nới lỏng điều kiện.

### 3.4. Kiểm tra sức khỏe

```http
GET /health   → {"status": "ok", "version": "1.0.0"}
GET /ready    → {"status": "ready", "models_loaded": true,
                 "car_count": 216, "class_count": 196}
```

`/ready` trả **503** khi chưa nạp được mô hình. Lúc đó chức năng tư vấn
**vẫn hoạt động** (chỉ cần bảng thông số), riêng nhận diện trả 503.

> Giao diện nên gọi `/ready` khi khởi động và vô hiệu hóa tab nhận diện
> nếu `models_loaded: false`.

---

## 4. Mã lỗi

| Mã | Khi nào | Thông báo cho người dùng |
| ---: | :--- | :--- |
| 400 | File rỗng hoặc ảnh hỏng | "File ảnh không hợp lệ" |
| 401 | Thiếu/sai API key | "Chưa đăng nhập" |
| 404 | Không tìm thấy dòng xe | "Không tìm thấy xe này" |
| 413 | File hoặc ảnh quá lớn | "Ảnh quá lớn, tối đa 10 MB" |
| 415 | Định dạng không hỗ trợ | "Chỉ hỗ trợ JPEG, PNG, BMP, GIF, WEBP" |
| 422 | Dữ liệu không hợp lệ | Đọc trường `detail` |
| 429 | Vượt hạn mức gọi | "Bạn thao tác quá nhanh, thử lại sau" |
| 503 | Mô hình chưa sẵn sàng | "Hệ thống đang khởi động" |

Mọi lỗi trả cùng khuôn dạng:

```json
{
  "detail": "Mô tả lỗi",
  "request_id": "a1b2c3d4e5f6"
}
```

`request_id` cũng nằm trong header `X-Request-ID`. Hiển thị mã này khi báo
lỗi để hỗ trợ tra log.

### Giới hạn tốc độ

| Endpoint | Hạn mức |
| :--- | :--- |
| `/api/v1/recognize` | 20 lần/phút |
| Các endpoint khác | 60 lần/phút |

Frontend nên chặn người dùng bấm liên tục (disable nút khi đang tải).

---

## 5. Yêu cầu giao diện

### Trang 1 — Nhận diện xe

#### Khu vực tải ảnh

- Kéo thả hoặc chọn file
- Dán URL ảnh (backend không hỗ trợ URL — frontend tự tải rồi gửi blob)
- Xem trước ảnh trước khi gửi
- Kiểm tra kích thước và định dạng **ở phía client** trước khi gửi, để
  người dùng biết sớm thay vì đợi server trả 413/415

#### Khu vực kết quả

- Vẽ khung bao lên ảnh (canvas hoặc SVG overlay), **quy đổi tọa độ** theo
  tỷ lệ hiển thị
- Mỗi xe: thanh xác suất top-5, thông số kỹ thuật, xe tương tự
- Ảnh nhiều xe: cho phép chọn từng xe, làm nổi khung tương ứng
- `is_confident: false` → hiển thị cảnh báo rõ ràng
- `likely_not_a_car: true` → thông báo có thể không phải ảnh ô tô

#### Trạng thái tải

Suy luận mất ~135 ms cho một xe, nhưng ảnh nhiều xe có thể lên 800 ms.
Hiển thị skeleton hoặc spinner.

### Trang 2 — Tư vấn xe

- Form: slider ngân sách (dùng `price_min`/`price_max` từ `/cars/filters`),
  chọn số chỗ, chọn nhiều kiểu dáng, slider mức tiêu hao
- Kết quả: danh sách xe kèm điểm phù hợp
- Trạng thái rỗng: gợi ý nới lỏng điều kiện cụ thể

### Trang 3 — Danh mục xe

- Bảng/lưới 216 dòng xe, phân trang
- Lọc theo hãng, kiểu dáng, phân khúc
- Bấm vào xe → trang chi tiết kèm xe tương tự

### Yêu cầu chung

| Hạng mục | Yêu cầu |
| :--- | :--- |
| Ngôn ngữ | Tiếng Việt (có dấu) |
| Responsive | Mobile-first, hoạt động tốt từ 360px |
| Chủ đề | Sáng/tối, theo `prefers-color-scheme` và cho phép chuyển |
| Khả năng tiếp cận | WCAG 2.2 AA: điều hướng bàn phím, ARIA labels, tương phản đủ |
| Định dạng giá | 594.4 → "594 triệu"; 1199.0 → "1,20 tỷ" |
| Xe điện | `fuel_l_per_100km: 0.0` → hiển thị "Xe điện", không phải "0.0 L" |

---

## 6. Cấu trúc thư mục đề xuất

```text
src/app/
├── core/
│   ├── api-types.ts            # Sinh từ openapi.json
│   ├── api.service.ts          # Gọi REST API
│   ├── api-key.interceptor.ts  # Thêm header X-API-Key
│   ├── error.interceptor.ts    # Chuyển mã lỗi thành thông báo
│   └── health.store.ts         # Trạng thái /ready
│
├── features/
│   ├── recognition/
│   │   ├── recognition.page.ts
│   │   ├── image-upload.component.ts
│   │   ├── detection-overlay.component.ts   # Vẽ khung bao
│   │   ├── prediction-list.component.ts
│   │   └── recognition.store.ts
│   ├── recommendation/
│   │   ├── recommendation.page.ts
│   │   ├── needs-form.component.ts
│   │   └── recommendation.store.ts
│   └── catalog/
│       ├── catalog.page.ts
│       ├── car-detail.page.ts
│       └── catalog.store.ts
│
├── shared/
│   ├── car-card.component.ts
│   ├── confidence-bar.component.ts
│   ├── price.pipe.ts           # 594.4 → "594 triệu"
│   ├── fuel.pipe.ts            # 0.0 → "Xe điện"
│   └── empty-state.component.ts
│
├── app.config.ts               # provideHttpClient, providers
└── app.routes.ts               # Lazy loading từng feature
```

---

## 7. Mẫu code

### Cấu hình ứng dụng

```typescript
// app.config.ts
import { ApplicationConfig, provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient, withFetch, withInterceptors } from '@angular/common/http';
import { provideRouter, withComponentInputBinding } from '@angular/router';

import { routes } from './app.routes';
import { apiKeyInterceptor } from './core/api-key.interceptor';
import { errorInterceptor } from './core/error.interceptor';

export const appConfig: ApplicationConfig = {
  providers: [
    provideZonelessChangeDetection(),
    provideRouter(routes, withComponentInputBinding()),
    provideHttpClient(
      withFetch(),
      withInterceptors([apiKeyInterceptor, errorInterceptor]),
    ),
  ],
};
```

### Service gọi API

```typescript
// core/api.service.ts
import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import { environment } from '../../environments/environment';
import type { components } from './api-types';

type RecognitionResponse = components['schemas']['RecognitionResponse'];
type RecommendRequest = components['schemas']['RecommendRequest'];
type RecommendResponse = components['schemas']['RecommendResponse'];

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = environment.apiBaseUrl;

  recognize(
    file: File,
    options: { topK?: number; includeSimilar?: boolean } = {},
  ): Observable<RecognitionResponse> {
    const form = new FormData();
    form.append('file', file);

    let params = new HttpParams();
    if (options.topK) {
      params = params.set('top_k', options.topK);
    }
    if (options.includeSimilar) {
      params = params.set('include_similar', true);
    }

    return this.http.post<RecognitionResponse>(
      `${this.baseUrl}/api/v1/recognize`,
      form,
      { params },
    );
  }

  recommend(needs: RecommendRequest): Observable<RecommendResponse> {
    return this.http.post<RecommendResponse>(
      `${this.baseUrl}/api/v1/recommend`,
      needs,
    );
  }
}
```

### Store bằng Signals

```typescript
// features/recognition/recognition.store.ts
import { computed, inject, signal } from '@angular/core';
import { signalStore, withComputed, withMethods, withState } from '@ngrx/signals';

import { ApiService } from '../../core/api.service';

type State = {
  result: RecognitionResponse | null;
  loading: boolean;
  error: string | null;
  selectedIndex: number;
};

export const RecognitionStore = signalStore(
  { providedIn: 'root' },
  withState<State>({
    result: null,
    loading: false,
    error: null,
    selectedIndex: 0,
  }),
  withComputed(({ result, selectedIndex }) => ({
    vehicles: computed(() => result()?.vehicles ?? []),
    selected: computed(() => result()?.vehicles[selectedIndex()] ?? null),
    // Cảnh báo khi nhiều khả năng ảnh không phải ô tô.
    notACar: computed(() => result()?.likely_not_a_car ?? false),
  })),
  withMethods((store, api = inject(ApiService)) => ({
    async recognize(file: File): Promise<void> {
      patchState(store, { loading: true, error: null });
      try {
        const result = await firstValueFrom(
          api.recognize(file, { topK: 5, includeSimilar: true }),
        );
        patchState(store, { result, loading: false, selectedIndex: 0 });
      } catch (error) {
        patchState(store, {
          error: toMessage(error),
          loading: false,
        });
      }
    },
  })),
);
```

### Template với cú pháp mới

```typescript
@Component({
  selector: 'app-prediction-list',
  standalone: true,
  imports: [MatProgressBarModule, PricePipe],
  template: `
    @if (vehicle().is_confident) {
      <h3>{{ vehicle().predictions[0].class_name }}</h3>
    } @else {
      <div class="warning" role="alert">
        Độ tin cậy chỉ
        {{ vehicle().predictions[0].confidence | percent:'1.1-1' }} —
        kết quả chỉ mang tính tham khảo.
      </div>
    }

    @for (prediction of vehicle().predictions; track prediction.class_name) {
      <div class="prediction-row">
        <span>{{ prediction.class_name }}</span>
        <mat-progress-bar
          mode="determinate"
          [value]="prediction.confidence * 100"
          [attr.aria-label]="prediction.class_name" />
        <span>{{ prediction.confidence | percent:'1.1-1' }}</span>
      </div>
    } @empty {
      <p>Không nhận diện được xe nào.</p>
    }
  `,
})
export class PredictionListComponent {
  readonly vehicle = input.required<DetectedVehicle>();
}
```

### Pipe định dạng giá

```typescript
// shared/price.pipe.ts
@Pipe({ name: 'price', standalone: true })
export class PricePipe implements PipeTransform {
  transform(millionVnd: number | null | undefined): string {
    if (millionVnd == null) {
      return '—';
    }
    // Từ 1000 triệu trở lên thì hiển thị theo tỷ cho dễ đọc.
    if (millionVnd >= 1000) {
      return `${(millionVnd / 1000).toFixed(2).replace('.', ',')} tỷ`;
    }
    return `${Math.round(millionVnd).toLocaleString('vi-VN')} triệu`;
  }
}
```

### Vẽ khung bao đúng tỷ lệ

```typescript
// features/recognition/detection-overlay.component.ts
//
// Tọa độ box theo ảnh GỐC, nhưng ảnh hiển thị thường bị thu nhỏ.
// Phải quy đổi, nếu không khung sẽ lệch.
@Component({
  selector: 'app-detection-overlay',
  standalone: true,
  template: `
    <div class="relative inline-block">
      <img #img [src]="imageUrl()" (load)="onLoad()" alt="Ảnh xe" />

      <svg class="absolute inset-0 w-full h-full pointer-events-none">
        @for (vehicle of vehicles(); track $index) {
          @if (vehicle.box; as box) {
            <rect
              [attr.x]="box.x1 * scale()"
              [attr.y]="box.y1 * scale()"
              [attr.width]="(box.x2 - box.x1) * scale()"
              [attr.height]="(box.y2 - box.y1) * scale()"
              [class.selected]="$index === selectedIndex()"
              fill="none" stroke-width="2" />
          }
        }
      </svg>
    </div>
  `,
})
export class DetectionOverlayComponent {
  readonly imageUrl = input.required<string>();
  readonly vehicles = input.required<DetectedVehicle[]>();
  readonly selectedIndex = input(0);

  private readonly imgRef = viewChild.required<ElementRef<HTMLImageElement>>('img');
  readonly scale = signal(1);

  onLoad(): void {
    const img = this.imgRef().nativeElement;
    // naturalWidth = kích thước gốc, clientWidth = kích thước hiển thị.
    this.scale.set(img.clientWidth / img.naturalWidth);
  }
}
```

---

## 8. Khởi tạo dự án

```bash
npx @angular/cli@latest new pvehicle-web \
  --style=scss --ssr=false --zoneless

cd pvehicle-web

ng add @angular/material
npm install @ngrx/signals
npm install -D tailwindcss @tailwindcss/postcss openapi-typescript vitest

# Sinh types từ API
npx openapi-typescript ../docs/frontend/openapi.json \
  -o src/app/core/api-types.ts
```

### Chạy backend để phát triển

```powershell
cd <thư mục PVehicle-AI>
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload
```

Tài liệu API tương tác: <http://localhost:8000/docs>

### CORS

Backend mặc định cho phép mọi origin (`PVEHICLE_CORS_ORIGINS=*`). Khi triển
khai thật phải ghi rõ tên miền của frontend.

---

## 9. Những điểm cần chú ý

**Đừng tin `content-type` khi kiểm tra file.** Backend kiểm tra chữ ký byte
đầu file. Frontend nên kiểm tra tương tự (đọc vài byte đầu bằng
`FileReader`) để báo lỗi sớm.

**Ảnh nhiều xe rất phổ biến.** Ảnh chụp đường phố thường có 3-5 xe. Giao
diện phải xử lý tốt trường hợp này, không chỉ hiển thị xe đầu tiên.

**Kích thước ảnh không giới hạn ở phía hiển thị.** Người dùng có thể tải
ảnh 4000×3000. Dùng `object-fit: contain` và giới hạn chiều cao.

**Xe điện có `fuel_l_per_100km: 0.0`.** Đừng hiển thị "0.0 L/100km" —
người dùng sẽ tưởng lỗi. Hiển thị "Xe điện".

**Giá là số liệu mô phỏng với xe quốc tế.** Backend sinh giá theo công
thức cho 196 dòng xe Stanford Cars. Riêng 20 dòng xe Việt Nam dùng giá
niêm yết thật. Giao diện nên có ghi chú nhỏ về điều này.

**`processing_ms` hữu ích cho việc gỡ lỗi.** Hiển thị ở chế độ dev hoặc
trong tooltip, không cần đưa lên giao diện chính.

---

## 10. Tiêu chí hoàn thành

- [ ] Tải ảnh và nhận kết quả nhận diện, có khung bao đúng vị trí
- [ ] Xử lý đúng ảnh nhiều xe (chọn từng xe)
- [ ] Xử lý `box: null` (không phát hiện được xe)
- [ ] Cảnh báo khi `is_confident: false` và `likely_not_a_car: true`
- [ ] Form tư vấn hoạt động, có trạng thái rỗng
- [ ] Danh mục xe có lọc và phân trang
- [ ] Mọi mã lỗi HTTP đều có thông báo tiếng Việt dễ hiểu
- [ ] Responsive từ 360px trở lên
- [ ] Chủ đề sáng/tối
- [ ] Điều hướng được hoàn toàn bằng bàn phím
- [ ] Kiểm thử: service, store, và các component chính
