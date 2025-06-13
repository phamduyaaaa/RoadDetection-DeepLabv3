import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Import InterpolationMode cho transforms.Resize (quan trọng cho các phiên bản torchvision mới)
from torchvision.transforms import InterpolationMode 

# Import mô hình DeepLabv3 từ file model.py của bạn
from model import DeepLabv3 

# --- Cấu hình cho việc dự đoán ---
# Đảm bảo các cấu hình này khớp với khi bạn huấn luyện mô hình
INPUT_SIZE = 256 # Kích thước ảnh đầu vào mà mô hình đã được huấn luyện
SAVE_MODEL_DIR = 'weight/model.pt' # Đường dẫn tới file model đã lưu sau khi huấn luyện
OUTPUT_PREDICTION_DIR = 'googlemap_output_images' # Thư mục để lưu các ảnh mask dự đoán mới

# --- Hàm dự đoán cho một ảnh đơn lẻ ---
def predict_single_image(model, image_path: str, device: torch.device):
    """
    Tải một ảnh, xử lý, chạy dự đoán bằng mô hình, và lưu mask kết quả.

    Args:
        model (nn.Module): Mô hình đã được huấn luyện.
        image_path (str): Đường dẫn tới file ảnh đầu vào.
        device (torch.device): Thiết bị (CPU/CUDA) để chạy dự đoán.
    """
    # Đảm bảo thư mục output_dir tồn tại
    os.makedirs(OUTPUT_PREDICTION_DIR, exist_ok=True)

    # 1. Tải và tiền xử lý ảnh đầu vào
    try:
        # Mở ảnh dưới dạng RGB (3 kênh)
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file ảnh tại đường dẫn: {image_path}")
        return
    except Exception as e:
        print(f"Lỗi khi mở hoặc xử lý ảnh '{image_path}': {e}")
        return

    # Định nghĩa Transform cho ảnh đầu vào (trong hàm để dễ quản lý hơn)
    # Phải khớp với transforms trong train.py
    # Kích thước ảnh PIL ban đầu có thể khác 256x256, sẽ được resize
    inference_transforms = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE), InterpolationMode.BILINEAR), 
        transforms.ToTensor() 
    ])

    # Áp dụng transform để có input_tensor
    input_tensor = inference_transforms(image).unsqueeze(0) # unsqueeze(0) để thêm chiều batch

    # --- DEBUG PRINTS ---
    print(f"\n--- Debugging Input Tensor ---")
    print(f"PIL Image (after initial open/convert): mode={image.mode}, size={image.size}")
    print(f"PIL Image (after resize for model input): size={(INPUT_SIZE, INPUT_SIZE)}") # Thông báo kích thước sau resize
    print(f"Input tensor shape BEFORE moving to device: {input_tensor.shape}")
    print(f"Input tensor dtype BEFORE moving to device: {input_tensor.dtype}")
    print(f"Input tensor min/max value BEFORE moving to device: {input_tensor.min().item():.4f}/{input_tensor.max().item():.4f}")
    # --- END DEBUG PRINTS ---

    # 2. Chuyển tensor sang thiết bị (GPU/CPU)
    input_tensor = input_tensor.to(device)

    # --- DEBUG PRINTS ---
    print(f"Input tensor dtype ON DEVICE: {input_tensor.dtype}")
    print(f"Input tensor device: {input_tensor.device}")
    print(f"Input tensor min/max value ON DEVICE: {input_tensor.min().item():.4f}/{input_tensor.max().item():.4f}")
    print(f"--- End Debugging ---")
    # --- END DEBUG PRINTS ---

    # 3. Chạy dự đoán
    model.eval() # Chuyển mô hình sang chế độ đánh giá
    with torch.no_grad(): # Tắt tính toán gradient để tiết kiệm bộ nhớ và tăng tốc
        output = model(input_tensor)

    # 4. Hậu xử lý output
    # DeepLabv3 thường trả về logits, nên áp dụng sigmoid để có xác suất
    # Mask của bạn trong dataset.py là torch.cat([y2, y1], dim=0) -> (2, H, W)
    # Kênh 1 tương ứng với đường (road), kênh 0 là nền (background)
    road_probabilities = torch.sigmoid(output[:, 1, :, :]).squeeze(0) # Lấy kênh đường, áp dụng sigmoid, loại bỏ chiều batch

    # Áp dụng ngưỡng để tạo mask nhị phân (0 hoặc 1)
    predicted_mask = (road_probabilities > 0.5).cpu().numpy() # Chuyển về CPU và numpy array

    # Chuyển đổi mask nhị phân thành ảnh PIL để lưu
    # Nhân với 255 để các pixel 1 (đường) trở thành 255 (trắng), pixel 0 (nền) vẫn là 0 (đen)
    predicted_mask_pil = Image.fromarray((predicted_mask * 255).astype(np.uint8))

    # 5. Lưu ảnh kết quả
    # Lấy tên file gốc của ảnh đầu vào (ví dụ: 'my_image.jpg')
    base_filename = os.path.basename(image_path)
    # Tạo tên file output (ví dụ: 'my_image_predicted.png')
    output_filename = f"{os.path.splitext(base_filename)[0]}_predicted.png"
    output_path = os.path.join(OUTPUT_PREDICTION_DIR, output_filename)
    predicted_mask_pil.save(output_path)

    print(f"\n--- Dự đoán hoàn tất ---")
    print(f"Đã lưu mask dự đoán cho '{base_filename}' tại: {output_path}")

    # (Tùy chọn) Hiển thị ảnh gốc và mask dự đoán
    # SỬA LỖI: Tạo ảnh đã resize để hiển thị
    resized_image_for_display = image.resize((INPUT_SIZE, INPUT_SIZE))
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Ảnh Gốc (resized cho model)")
    # Sử dụng biến vừa tạo
    plt.imshow(resized_image_for_display)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Mask Dự Đoán")
    plt.imshow(predicted_mask_pil, cmap='gray')
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    # --- Cấu hình thiết bị ---
    # Ưu tiên GPU (cuda), sau đó là MPS (Apple Silicon), cuối cùng là CPU
    DEVICE = 'cpu'
    if torch.cuda.is_available():
        DEVICE = 'cuda:0'
    elif torch.backends.mps.is_available(): 
        DEVICE = 'mps'
    print(f"Sử dụng thiết bị: {DEVICE}")
    
    # --- Tải mô hình đã huấn luyện ---
    # Khởi tạo mô hình DeepLabv3 (không truyền num_classes nếu model.py không nhận)
    model = DeepLabv3() 
    
    # Kiểm tra sự tồn tại của file model.pt trước khi tải
    if not os.path.exists(SAVE_MODEL_DIR):
        print(f"Lỗi: Không tìm thấy file model đã lưu tại đường dẫn: {SAVE_MODEL_DIR}")
        print("Vui lòng chạy script 'train.py' trước để huấn luyện và lưu mô hình.")
        exit()

    # Tải checkpoint và nạp trọng số vào mô hình
    # map_location đảm bảo trọng số được tải lên đúng thiết bị
    checkpoint = torch.load(SAVE_MODEL_DIR, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE) # Chuyển mô hình sang thiết bị để dự đoán

    print("Mô hình đã được tải thành công từ checkpoint.")

    # --- Yêu cầu người dùng nhập đường dẫn ảnh ---
    print("\n--- Nhập đường dẫn ảnh để dự đoán ---")
    print("Ví dụ: dataset/Test/image/1.bmp hoặc /path/to/your/image.jpg")
    input_image_path = input("Nhập đường dẫn đến ảnh đầu vào: ")

    # --- Chạy dự đoán cho ảnh đã nhập ---
    # Truyền OUTPUT_PREDICTION_DIR vào hàm
    predict_single_image(model, input_image_path, DEVICE)
