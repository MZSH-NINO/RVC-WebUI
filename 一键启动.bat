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

echo [2/2] 启动 WebUI...
echo.

REM 检测端口是否已被占用
netstat -an | find ":7865" | find "LISTENING" >nul 2>&1
if %errorlevel% equ 0 (
    echo [警告] 端口 7865 已被占用，可能有其他 WebUI 实例正在运行
    echo [提示] 请先关闭其他实例，或在任务管理器中结束 python.exe 进程
    echo.
    pause
    exit /b 1
)

echo [信息] 正在加载模型，请稍候...
echo [提示] 首次运行可能需要下载额外模型
echo [提示] 浏览器将在 WebUI 启动后自动打开
echo [提示] 请保持此窗口打开，关闭窗口将停止 WebUI 服务
echo.

REM 在当前窗口启动 webui（前台阻塞运行）
python infer-web.py
set WEBUI_EXIT_CODE=%errorlevel%

echo.
echo ========================================
echo WebUI 已停止运行 (退出代码: %WEBUI_EXIT_CODE%)
echo ========================================
echo.
pause
