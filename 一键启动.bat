@echo off
chcp 65001 >nul

echo ========================================
echo RVC-WebUI 启动脚本
echo ========================================
echo.

REM 设置 HuggingFace 镜像（国内用户推荐）
set HF_ENDPOINT=https://hf-mirror.com

REM 自动添加 ffmpeg 到 PATH
if exist "ffmpeg.exe" (
    set "PATH=%cd%;%PATH%"
    echo [✓] 已添加项目内 ffmpeg 到 PATH
)

REM 激活 Conda 环境
echo [1/2] 激活 Conda 环境...
call conda activate RVC-WebUI
if %errorlevel% neq 0 (
    echo.
    echo [错误] 无法激活 RVC-WebUI 环境
    echo [提示] 请先运行"一键部署工具\一键部署.bat"进行部署
    echo.
    pause
    exit /b 1
)

echo [2/2] 启动 Simple GUI...
echo.
echo [信息] 正在加载简易桌面界面...
echo [提示] 请保持此窗口打开，关闭窗口将停止 GUI
echo [提示] 如需使用原始 Gradio WebUI，请运行 go-web.bat
echo.

python simple_gui.py
set WEBUI_EXIT_CODE=%errorlevel%

echo.
echo ========================================
echo 程序已停止运行 (退出代码: %WEBUI_EXIT_CODE%)
echo ========================================
echo.
pause
