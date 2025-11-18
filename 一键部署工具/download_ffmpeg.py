"""
RVC-WebUI ffmpeg 自动下载脚本
"""
import os
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
RVC_DOWNLOAD_LINK = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"


def download_file(url, filename):
    """下载文件"""
    print(f"正在下载 {filename}...")
    try:
        r = requests.get(url, timeout=300)
        r.raise_for_status()

        target_path = BASE_DIR / filename
        with open(target_path, 'wb') as f:
            f.write(r.content)

        print(f"✓ {filename} 下载成功！")
        return True
    except Exception as e:
        print(f"✗ {filename} 下载失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("RVC-WebUI ffmpeg 下载工具")
    print("=" * 60)
    print()

    # 检查是否已存在
    ffmpeg_path = BASE_DIR / "ffmpeg.exe"
    ffprobe_path = BASE_DIR / "ffprobe.exe"

    if ffmpeg_path.exists() and ffprobe_path.exists():
        print("[信息] ffmpeg 和 ffprobe 已存在，跳过下载")
        return 0

    # 下载 ffmpeg
    success = True
    if not ffmpeg_path.exists():
        success = download_file(RVC_DOWNLOAD_LINK + "ffmpeg.exe", "ffmpeg.exe") and success
    else:
        print("[✓] ffmpeg.exe 已存在")

    # 下载 ffprobe
    if not ffprobe_path.exists():
        success = download_file(RVC_DOWNLOAD_LINK + "ffprobe.exe", "ffprobe.exe") and success
    else:
        print("[✓] ffprobe.exe 已存在")

    print()
    if success:
        print("[✓] 所有文件下载完成！")
        return 0
    else:
        print("[⚠️] 部分文件下载失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
