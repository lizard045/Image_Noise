# 1. 僅使用傳統方法（原有功能）
python taipei101_denoise.py --input_dir testsets/taipei101 --output_dir taipei101_traditional

# 2. 使用 Stable Diffusion 增強
python taipei101_denoise.py --input_dir testsets/taipei101 --output_dir taipei101_sd --use_sd --sd_strength 0.3

# 3. 使用混合增強模式（推薦）
python taipei101_denoise.py --input_dir testsets/taipei101 --output_dir taipei101_hybrid --hybrid_mode

# 4. 自訂SD模型和參數
python taipei101_denoise.py --input_dir testsets/taipei101 --output_dir taipei101_custom --use_sd --sd_model "stabilityai/stable-diffusion-2-1" --sd_strength 0.4