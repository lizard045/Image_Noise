#!/usr/bin/env python3
"""
進階影像去噪演示程式
執行這個腳本來運行所有的進階去噪方法並查看結果
"""

import subprocess
import sys
import os

def main():
    print("🎯 進階影像去噪演示程式")
    print("=" * 60)
    
    # 檢查依賴套件
    print("📦 檢查並安裝依賴套件...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ 依賴套件已安裝")
    except subprocess.CalledProcessError as e:
        print(f"❌ 安裝依賴套件失敗: {e}")
        return
    
    # 檢查影像檔案
    if not os.path.exists('taipei101.png'):
        print("❌ 找不到 taipei101.png 檔案")
        return
    
    print("✅ 影像檔案存在")
    
    # 執行進階去噪程式
    print("\n🚀 開始執行進階去噪程式...")
    print("⏰ 這可能需要幾分鐘時間，請耐心等待...")
    
    try:
        result = subprocess.run([sys.executable, "advanced_denoising.py"], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("⚠️  警告訊息:")
            print(result.stderr)
    except Exception as e:
        print(f"❌ 執行失敗: {e}")
        return
    
    # 顯示結果檔案
    print("\n📂 查看結果檔案...")
    results_dir = "advanced_denoised_results"
    if os.path.exists(results_dir):
        files = [f for f in os.listdir(results_dir) if f.endswith('.png')]
        files.sort()
        print(f"📁 結果資料夾: {results_dir}")
        for i, file in enumerate(files, 1):
            print(f"  {i:2d}. {file}")
    
    print("\n🎉 演示完成！")
    print("💡 提示：")
    print("   - 查看 'advanced_denoised_results' 資料夾中的結果影像")
    print("   - 對比各種方法的PSNR和SSIM指標")
    print("   - 這些進階方法應該比階段一有更好的去噪效果！")

if __name__ == "__main__":
    main() 