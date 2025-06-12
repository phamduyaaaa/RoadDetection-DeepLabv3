# 🛣️ Road Detection sử dụng DeepLabv3 trên ảnh vệ tinh

### 📌 Tổng quan
#### Dự án này triển khai mô hình phân đoạn semantic sử dụng DeepLabv3 với backbone ResNet101 để nhận diện đường giao thông trong ảnh vệ tinh. Mô hình được huấn luyện trên tập dữ liệu TGRS Road, gồm các cặp ảnh RGB và mặt nạ nhãn tương ứng. Mục tiêu là xác định chính xác khu vực là "đường" trên ảnh vệ tinh độ phân giải cao.

### 🏗 Cấu trúc Dự án
```bash
Python 3.10.18
RoadDetection-DeepLabv3/
├── dataset/                # Dataset folders (Train / Validation / Test)
│   ├── Train/
│   ├── Validation/
│   └── Test/
│
├── dataset.py              # Dataset class & transforms
├── metrics.py              # Accuracy, IoU, and other metrics
├── model.py                # DeepLabv3 model with ResNet101 backbone
├── train.py                # Main training script
├── utils.py                # Visualization and utility functions
│
├── output_images/          # Output prediction visualizations
│   ├── figure_0_0.png
│   ├── figure_0_1.png
│   ├── ...
│   └── figure_1_3.png
│
├── plot/                   # Training logs and plots
│   ├── training_metrics_plot.png
│   ├── train_losses.npy
│   ├── val_losses.npy
│   ├── train_accuracies.npy
│   ├── val_accuracies.npy
│   ├── train_ious.npy
│   └── val_ious.npy
│
├── weight/                 # Saved model weights
│   └── model.pt
│
├── __pycache__/            # Python bytecode cache (auto-generated)
```
### 🛰️ Tập Dữ liệu: TGRS Road
#### Tập dữ liệu [TGRS Road](https://www.kaggle.com/datasets/ipythonx/tgrs-road) được thu thập từ ảnh vệ tinh độ phân giải cao, cung cấp các cặp ảnh đầu vào và mặt nạ phân đoạn. Tập dữ liệu bao gồm các khu vực đô thị, nông thôn với độ đa dạng cao về địa hình và cấu trúc đường xá.


### 🧠 Kiến trúc DeepLabv3
#### DeepLabv3 sử dụng atrous convolution để mở rộng receptive field mà không làm mất thông tin độ phân giải không gian. Cụ thể, mô hình trong dự án này gồm:
```bash
├── Backbone: ResNet101
├── Atrous Spatial Pyramid Pooling (ASPP)
├── Output stride: 16
├── Segmentation head: 1x1 Conv → Upsampling
├── Loss: CrossEntropyLoss
```
### 📊 Kết quả
![training_metrics_plot](https://github.com/user-attachments/assets/f1f4c5a0-8c4e-4bae-ba5b-6de3f3573a2f)

### 🖼 Một số kết quả đầu ra (output_images)
<p align="center">
  <img src="https://github.com/user-attachments/assets/5d0e4a98-56b2-4061-b561-768664362a9d" width="250"/>
  <img src="https://github.com/user-attachments/assets/4f063c9d-89b0-46b6-b124-6b47291326ea" width="250"/>
  <img src="https://github.com/user-attachments/assets/35f2767f-5396-4796-ab85-b20a14280fdc" width="250"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/f5fa13db-ba1f-44cb-ae20-a8f3d270322c" width="250"/>
  <img src="https://github.com/user-attachments/assets/962c8417-7672-4c34-b02a-078c81dc4c67" width="250"/>
  <img src="https://github.com/user-attachments/assets/2aec4590-6395-4c85-a654-b4bea0e2cbda" width="250"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/cade1c65-ed31-4962-b7c9-de0df1e59159" width="250"/>
  <img src="https://github.com/user-attachments/assets/1264c1a7-b13b-4a1c-bd06-d08667c7f5dd" width="250"/>
  <img src="https://github.com/user-attachments/assets/3d0fa2bd-270c-4e8c-850c-1faf630f77eb" width="250"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/b18fb61f-4ce0-4683-8a73-f42f4da8f132" width="250"/>
  <img src="https://github.com/user-attachments/assets/b8d34a55-1de1-46d0-aad4-8a5961b9eb23" width="250"/>
  <img src="https://github.com/user-attachments/assets/097184a9-182d-4580-aa14-b76b09c8a67e" width="250"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/50132401-c710-4bfa-b407-66659f9c9328" width="250"/>
  <img src="https://github.com/user-attachments/assets/0d93deff-41c4-487d-8f00-8b96afc5ad4d" width="250"/>
  <img src="https://github.com/user-attachments/assets/c6328ee7-ba31-420d-8e8e-ecd35467e9d6" width="250"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/f3b3c01e-8fff-492b-ba89-cd895facaa1b" width="250"/>
  <img src="https://github.com/user-attachments/assets/0a09731e-5ad4-4378-857d-9dabc0b02d0a" width="250"/>
  <img src="https://github.com/user-attachments/assets/0ee83c45-d102-486e-8547-6c25bf92302b" width="250"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/86cd92a5-abc3-4617-a931-7024a1a5f3cf" width="250"/>
  <img src="https://github.com/user-attachments/assets/adfb2614-560d-4fbe-b9e1-13c8dd05972f" width="250"/>
</p>








