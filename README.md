# Model Lip Sync siêu thật video 4k độ dài không giới hạn

<table>
<tr>
<td width="50%">

![](assets/thl.PNG)

</td>
<td width="50%">

https://github.com/user-attachments/assets/your-video-id-here

*Upload `output/video_out.mp4` as an attachment to get the GitHub asset URL*

</td>
</tr>
</table>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2412.09262)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/ByteDance/LatentSync-1.6)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/fffiloni/LatentSync)
<a href="https://replicate.com/lucataco/latentsync"><img src="https://replicate.com/lucataco/latentsync/badge" alt="Replicate"></a>

*Audio-conditioned lip-sync chất lượng cao*

</div>

---

## 📖 Giới thiệu

**LatentSync** là một phương pháp lip-sync end-to-end dựa trên audio-conditioned latent diffusion models, không sử dụng bất kỳ biểu diễn chuyển động trung gian nào. Framework này tận dụng khả năng mạnh mẽ của Stable Diffusion để mô hình hóa trực tiếp mối tương quan phức tạp giữa âm thanh và hình ảnh.

### ✨ Tính năng nổi bật

- 🎭 **Chất lượng lip-sync cao**: Đồng bộ môi tự nhiên và chính xác
- 🚀 **Dễ dàng sử dụng**: Hỗ trợ cả Gradio app và command line
- 🔧 **Linh hoạt**: Điều chỉnh các tham số để tối ưu kết quả
- 💯 **Hiệu suất tốt**: Hỗ trợ nhiều độ phân giải và tối ưu VRAM

### 🖥️ Yêu cầu hệ thống

- **Python**: 3.10
- **CUDA**: 12.4 (khuyến nghị cho GPU acceleration)
- **GPU VRAM**: 
  - LatentSync 1.5: Tối thiểu 8GB
  - LatentSync 1.6: Tối thiểu 18GB

## 🔧 Cài đặt

### 1. Tạo môi trường ảo
```bash
# Linux/Mac
python3.10 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Cài đặt dependencies và tải checkpoints
```bash
source setup_env.sh
```

### 3. Kiểm tra cấu trúc checkpoints
Sau khi cài đặt thành công, checkpoints sẽ có cấu trúc như sau:

```
./checkpoints/
|-- latentsync_unet.pt
|-- whisper
|   `-- tiny.pt
```

**Lưu ý:** Bạn cũng có thể tải `latentsync_unet.pt` và `tiny.pt` thủ công từ [HuggingFace repo](https://huggingface.co/ByteDance/LatentSync-1.6)

## 🚀 Hướng dẫn sử dụng

### Phương pháp 1: Gradio App (Giao diện đồ họa)

Chạy ứng dụng Gradio để sử dụng giao diện đồ họa:

```bash
python gradio_app.py
```

### Phương pháp 2: Command Line Interface

#### Sử dụng script có sẵn:
```bash
./inference.sh
```

#### Chạy trực tiếp với một dòng lệnh:
```bash
python -m scripts.inference --unet_config_path "configs/unet/stage2_512.yaml" --inference_ckpt_path "checkpoints/latentsync_unet.pt" --inference_steps 20 --guidance_scale 1.5 --enable_deepcache --video_path "assets/thl2_trimmed.mp4" --audio_path "assets/thl_trimmed.wav" --video_out_path "video_out.mp4"
```

### ⚙️ Tùy chỉnh tham số inference

Điều chỉnh các tham số sau để đạt kết quả tốt nhất:

- **`inference_steps`** [20-50]: Giá trị cao hơn cải thiện chất lượng hình ảnh nhưng chậm hơn
- **`guidance_scale`** [1.0-3.0]: Giá trị cao hơn cải thiện độ chính xác lip-sync nhưng có thể gây méo hoặc giật hình
- **`enable_deepcache`**: Bật để tăng tốc độ inference

## 🔄 Data Processing Pipeline

Pipeline xử lý dữ liệu hoàn chỉnh bao gồm các bước sau:

1. Loại bỏ các file video bị lỗi
2. Resample video FPS về 25, và resample audio về 16000 Hz
3. Phát hiện cảnh qua [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)
4. Chia mỗi video thành các đoạn 5-10 giây
5. Affine transform khuôn mặt dựa trên landmarks từ [InsightFace](https://github.com/deepinsight/insightface), resize về 256×256
6. Loại bỏ video có [sync confidence score](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf) thấp hơn 3, điều chỉnh audio-visual offset về 0
7. Tính [hyperIQA](https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.pdf) score, loại bỏ video có điểm thấp hơn 40

Chạy script để thực thi pipeline:

```bash
./data_processing_pipeline.sh
```

**Lưu ý:** Thay đổi tham số `input_dir` trong script để chỉ định thư mục dữ liệu cần xử lý.

## 🏋️‍♂️ Training

### Training U-Net

#### Chuẩn bị

Trước khi training, bạn cần xử lý dữ liệu như mô tả ở phần trên. Tải pretrained SyncNet checkpoint:

```bash
huggingface-cli download ByteDance/LatentSync-1.6 stable_syncnet.pt --local-dir checkpoints
```

#### Bắt đầu training

```bash
./train_unet.sh
```

#### Các config file có sẵn

Thư mục `configs/unet` chứa nhiều file cấu hình:

- **`stage1.yaml`**: Stage1 training, yêu cầu **23 GB** VRAM
- **`stage2.yaml`**: Stage2 training hiệu suất tối ưu, yêu cầu **30 GB** VRAM
- **`stage2_efficient.yaml`**: Stage2 hiệu quả, yêu cầu **20 GB** VRAM (phù hợp RTX 3090)
- **`stage1_512.yaml`**: Stage1 với độ phân giải 512×512, yêu cầu **30 GB** VRAM
- **`stage2_512.yaml`**: Stage2 với độ phân giải 512×512, yêu cầu **55 GB** VRAM

#### Tạo danh sách file dữ liệu

```bash
python -m tools.write_fileslist
```

### Training SyncNet

Nếu muốn train SyncNet trên dataset riêng:

```bash
./train_syncnet.sh
```

## 📊 Evaluation

### Đánh giá sync confidence score

```bash
./eval/eval_sync_conf.sh
```

### Đánh giá độ chính xác SyncNet

```bash
./eval/eval_syncnet_acc.sh
```

**Lưu ý:** Dữ liệu test cần được xử lý qua pipeline trước khi đánh giá.

## 🔥 Updates

- `2025/06/11`: Phát hành **LatentSync 1.6** - train trên video 512×512 để giảm độ mờ. Xem demo [tại đây](docs/changelog_v1.6.md)
- `2025/03/14`: Phát hành **LatentSync 1.5** - cải thiện tính nhất quán thời gian, hiệu suất trên video tiếng Trung, và giảm VRAM xuống **20 GB**. Chi tiết [tại đây](docs/changelog_v1.5.md)

## 📑 Open-source Plan

- [x] Inference code và checkpoints
- [x] Data processing pipeline
- [x] Training code

## 🙏 Acknowledgement

- Our code is built on [AnimateDiff](https://github.com/guoyww/AnimateDiff). 
- Some code are borrowed from [MuseTalk](https://github.com/TMElyralab/MuseTalk), [StyleSync](https://github.com/guanjz20/StyleSync), [SyncNet](https://github.com/joonson/syncnet_python), [Wav2Lip](https://github.com/Rudrabha/Wav2Lip).

Thanks for their generous contributions to the open-source community!

## 📖 Citation

If you find our repo useful for your research, please consider citing our paper:

```bibtex
@article{li2024latentsync,
  title={LatentSync: Taming Audio-Conditioned Latent Diffusion Models for Lip Sync with SyncNet Supervision},
  author={Li, Chunyu and Zhang, Chao and Xu, Weikai and Lin, Jingyu and Xie, Jinghui and Feng, Weiguo and Peng, Bingyue and Chen, Cunjian and Xing, Weiwei},
  journal={arXiv preprint arXiv:2412.09262},
  year={2024}
}
```
