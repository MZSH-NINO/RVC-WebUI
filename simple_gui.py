import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
from pathlib import Path
from dotenv import load_dotenv

# 初始化路径
now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()

from configs.config import Config
from infer.modules.vc.modules import VC
from scipy.io import wavfile
import subprocess
import json
from random import shuffle
import numpy as np
import faiss
from sklearn.cluster import MiniBatchKMeans


class SimpleRVCGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RVC 简易工具 - 训练与推理")
        self.root.geometry("950x750")

        # 初始化配置
        self.config = Config()
        self.vc = VC(self.config)

        # 训练进度标志
        self.training_in_progress = False

        # 模型数据（用于记录模型来源和路径）
        self.model_data = []

        # 创建主布局
        self.create_widgets()

        # 加载模型列表
        self.refresh_models()

    def create_widgets(self):
        # 创建 Notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建两个标签页
        self.inference_frame = ttk.Frame(self.notebook)
        self.train_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.inference_frame, text="音频转换")
        self.notebook.add(self.train_frame, text="模型训练")

        # 创建推理界面
        self.create_inference_ui()

        # 创建训练界面
        self.create_training_ui()

    def create_inference_ui(self):
        """创建推理界面"""
        frame = self.inference_frame

        # 模型选择
        ttk.Label(frame, text="模型选择", font=("微软雅黑", 10, "bold")).grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)

        model_frame = ttk.Frame(frame)
        model_frame.grid(row=1, column=0, columnspan=3, sticky=tk.EW, padx=10, pady=5)

        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, state="readonly", width=50)
        self.model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(model_frame, text="刷新", command=self.refresh_models).pack(side=tk.LEFT, padx=5)
        ttk.Button(model_frame, text="加载模型", command=self.load_model).pack(side=tk.LEFT)

        # 输入音频
        ttk.Label(frame, text="输入音频", font=("微软雅黑", 10, "bold")).grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)

        input_frame = ttk.Frame(frame)
        input_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, padx=10, pady=5)

        self.input_path_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.input_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(input_frame, text="浏览", command=self.browse_input).pack(side=tk.LEFT, padx=5)

        # 输出音频
        ttk.Label(frame, text="输出音频", font=("微软雅黑", 10, "bold")).grid(row=4, column=0, sticky=tk.W, padx=10, pady=5)

        output_frame = ttk.Frame(frame)
        output_frame.grid(row=5, column=0, columnspan=3, sticky=tk.EW, padx=10, pady=5)

        self.output_path_var = tk.StringVar(value="output.wav")
        ttk.Entry(output_frame, textvariable=self.output_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="浏览", command=self.browse_output).pack(side=tk.LEFT, padx=5)

        # 参数设置
        ttk.Label(frame, text="参数设置", font=("微软雅黑", 10, "bold")).grid(row=6, column=0, sticky=tk.W, padx=10, pady=5)

        param_frame = ttk.Frame(frame)
        param_frame.grid(row=7, column=0, columnspan=3, sticky=tk.EW, padx=10, pady=5)

        # 音高调整
        ttk.Label(param_frame, text="音高调整:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.f0_up_key = tk.IntVar(value=0)
        ttk.Spinbox(param_frame, from_=-12, to=12, textvariable=self.f0_up_key, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        # 音高提取算法
        ttk.Label(param_frame, text="音高提取算法:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.f0_method = tk.StringVar(value="harvest")
        ttk.Combobox(param_frame, textvariable=self.f0_method, values=["pm", "harvest", "crepe", "rmvpe"], state="readonly", width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # 索引率
        ttk.Label(param_frame, text="索引率:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.index_rate = tk.DoubleVar(value=0.66)
        ttk.Scale(param_frame, from_=0, to=1, variable=self.index_rate, orient=tk.HORIZONTAL).grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Label(param_frame, textvariable=self.index_rate).grid(row=2, column=2, sticky=tk.W, padx=5, pady=2)

        # 滤波半径
        ttk.Label(param_frame, text="滤波半径:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.filter_radius = tk.IntVar(value=3)
        ttk.Spinbox(param_frame, from_=0, to=7, textvariable=self.filter_radius, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)

        # 保护清辅音
        ttk.Label(param_frame, text="保护清辅音:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.protect = tk.DoubleVar(value=0.33)
        ttk.Scale(param_frame, from_=0, to=0.5, variable=self.protect, orient=tk.HORIZONTAL).grid(row=4, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Label(param_frame, textvariable=self.protect).grid(row=4, column=2, sticky=tk.W, padx=5, pady=2)

        # 转换按钮
        ttk.Button(frame, text="开始转换", command=self.convert_audio).grid(row=8, column=0, columnspan=3, pady=10)

        # 日志输出
        ttk.Label(frame, text="日志输出", font=("微软雅黑", 10, "bold")).grid(row=9, column=0, sticky=tk.W, padx=10, pady=5)

        self.inference_log = scrolledtext.ScrolledText(frame, height=10, wrap=tk.WORD)
        self.inference_log.grid(row=10, column=0, columnspan=3, sticky=tk.NSEW, padx=10, pady=5)

        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(10, weight=1)

    def create_training_ui(self):
        """创建训练界面"""
        frame = self.train_frame

        # 实验名称
        ttk.Label(frame, text="实验名称", font=("微软雅黑", 10, "bold")).grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.exp_name_var = tk.StringVar(value="my_model")
        ttk.Entry(frame, textvariable=self.exp_name_var, width=50).grid(row=1, column=0, columnspan=3, sticky=tk.EW, padx=10, pady=5)

        # 数据集目录
        ttk.Label(frame, text="数据集目录", font=("微软雅黑", 10, "bold")).grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)

        dataset_frame = ttk.Frame(frame)
        dataset_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, padx=10, pady=5)

        self.dataset_path_var = tk.StringVar()
        ttk.Entry(dataset_frame, textvariable=self.dataset_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dataset_frame, text="浏览", command=self.browse_dataset).pack(side=tk.LEFT, padx=5)

        # 训练参数
        ttk.Label(frame, text="训练参数", font=("微软雅黑", 10, "bold")).grid(row=4, column=0, sticky=tk.W, padx=10, pady=5)

        param_frame = ttk.Frame(frame)
        param_frame.grid(row=5, column=0, columnspan=3, sticky=tk.EW, padx=10, pady=5)

        # 采样率
        ttk.Label(param_frame, text="采样率:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.sr_var = tk.StringVar(value="40k")
        ttk.Combobox(param_frame, textvariable=self.sr_var, values=["32k", "40k", "48k"], state="readonly", width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        # 是否使用音高
        self.if_f0_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="使用音高引导", variable=self.if_f0_var).grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)

        # 音高提取算法
        ttk.Label(param_frame, text="音高提取算法:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.train_f0_method = tk.StringVar(value="harvest")
        ttk.Combobox(param_frame, textvariable=self.train_f0_method, values=["pm", "harvest", "crepe", "rmvpe"], state="readonly", width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # 训练轮数
        ttk.Label(param_frame, text="训练轮数:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.total_epoch = tk.IntVar(value=200)
        ttk.Spinbox(param_frame, from_=10, to=1000, textvariable=self.total_epoch, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        # 批次大小
        ttk.Label(param_frame, text="批次大小:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.batch_size = tk.IntVar(value=8)
        ttk.Spinbox(param_frame, from_=1, to=32, textvariable=self.batch_size, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)

        # 保存频率
        ttk.Label(param_frame, text="保存频率:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.save_epoch = tk.IntVar(value=10)
        ttk.Spinbox(param_frame, from_=1, to=100, textvariable=self.save_epoch, width=10).grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)

        # 一键训练按钮
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=6, column=0, columnspan=3, sticky=tk.EW, padx=10, pady=10)

        ttk.Button(btn_frame, text="开始训练", command=self.auto_train, style="Accent.TButton").pack(pady=5)

        # 训练日志
        ttk.Label(frame, text="训练日志", font=("微软雅黑", 10, "bold")).grid(row=8, column=0, sticky=tk.W, padx=10, pady=5)

        self.training_log = scrolledtext.ScrolledText(frame, height=15, wrap=tk.WORD)
        self.training_log.grid(row=9, column=0, columnspan=3, sticky=tk.NSEW, padx=10, pady=5)

        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(9, weight=1)

    def refresh_models(self):
        """刷新模型列表（仅显示成品模型）"""
        models = []

        # 扫描 assets/weights 目录（成品模型）
        weight_root = os.getenv("weight_root", "assets/weights")
        if os.path.exists(weight_root):
            for name in os.listdir(weight_root):
                if name.endswith(".pth"):
                    models.append(name)

        self.model_combo['values'] = models
        self.model_data = models  # 保存模型列表

        if models:
            self.model_combo.current(0)

    def load_model(self):
        """加载模型"""
        model_name = self.model_var.get()
        if not model_name:
            messagebox.showwarning("警告", "请先选择一个模型！")
            return

        def load():
            try:
                self.log_inference(f"正在加载模型: {model_name}...")
                self.vc.get_vc(model_name)
                self.log_inference(f"模型加载成功！")
                messagebox.showinfo("成功", f"模型加载成功！")
            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                self.log_inference(f"模型加载错误:\n{error_msg}")
                messagebox.showerror("错误", f"模型加载失败: {str(e)}")

        threading.Thread(target=load, daemon=True).start()

    def browse_input(self):
        """选择输入音频"""
        filename = filedialog.askopenfilename(
            title="选择输入音频",
            filetypes=[("音频文件", "*.wav *.mp3 *.flac *.m4a"), ("所有文件", "*.*")]
        )
        if filename:
            self.input_path_var.set(filename)

    def browse_output(self):
        """选择输出路径"""
        filename = filedialog.asksaveasfilename(
            title="保存输出音频",
            defaultextension=".wav",
            filetypes=[("WAV 文件", "*.wav"), ("所有文件", "*.*")]
        )
        if filename:
            self.output_path_var.set(filename)

    def browse_dataset(self):
        """选择数据集目录"""
        dirname = filedialog.askdirectory(title="选择数据集目录")
        if dirname:
            self.dataset_path_var.set(dirname)

    def convert_audio(self):
        """转换音频"""
        input_path = self.input_path_var.get()
        output_path = self.output_path_var.get()

        if not input_path or not os.path.exists(input_path):
            messagebox.showwarning("警告", "请选择一个有效的输入音频文件！")
            return

        if not output_path:
            messagebox.showwarning("警告", "请指定输出路径！")
            return

        def convert():
            try:
                self.log_inference("开始音频转换...")
                self.log_inference(f"输入: {input_path}")
                self.log_inference(f"参数: 音高={self.f0_up_key.get()}, 算法={self.f0_method.get()}")

                # 自动查找对应的 index 文件
                model_name = self.model_var.get()
                index_file = ""

                if model_name:
                    # 成品模型，index 在 logs/模型名（不含.pth） 下
                    model_base = model_name.replace(".pth", "")
                    index_dir = os.path.join("logs", model_base)

                    # 尝试查找 index 文件
                    if os.path.exists(index_dir):
                        for file in os.listdir(index_dir):
                            if file.endswith(".index") and "added" in file:
                                index_file = os.path.join(index_dir, file)
                                break

                if index_file:
                    self.log_inference(f"使用索引文件: {index_file}")
                else:
                    self.log_inference("未找到索引文件，不使用索引")

                # 执行推理
                info, wav_opt = self.vc.vc_single(
                    0,  # sid
                    input_path,
                    self.f0_up_key.get(),
                    None,  # f0_file
                    self.f0_method.get(),
                    index_file,
                    "",  # file_index2
                    self.index_rate.get(),
                    self.filter_radius.get(),
                    0,  # resample_sr
                    1,  # rms_mix_rate
                    self.protect.get(),
                )

                self.log_inference(f"推理结果: {info}")

                if wav_opt is not None and len(wav_opt) == 2:
                    sr, audio = wav_opt
                    wavfile.write(output_path, sr, audio)
                    self.log_inference(f"音频已保存至: {output_path}")
                    messagebox.showinfo("成功", "音频转换完成！")
                else:
                    self.log_inference("错误: 输出格式无效")
                    messagebox.showerror("错误", "音频转换失败！")

            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                self.log_inference(f"错误: {error_msg}")
                messagebox.showerror("错误", f"转换失败: {str(e)}")

        threading.Thread(target=convert, daemon=True).start()

    def preprocess_dataset(self):
        """预处理数据集"""
        exp_name = self.exp_name_var.get()
        dataset_path = self.dataset_path_var.get()

        if not exp_name:
            messagebox.showwarning("警告", "请输入实验名称！")
            return

        if not dataset_path or not os.path.exists(dataset_path):
            messagebox.showwarning("警告", "请选择一个有效的数据集目录！")
            return

        def preprocess():
            try:
                self.log_training(f"预处理数据集: {dataset_path}")
                self.log_training(f"实验名称: {exp_name}")

                sr_dict = {"32k": 32000, "40k": 40000, "48k": 48000}
                sr = sr_dict[self.sr_var.get()]

                exp_dir = os.path.join(now_dir, "logs", exp_name)
                os.makedirs(exp_dir, exist_ok=True)

                # 调用预处理脚本
                python_cmd = sys.executable
                n_p = os.cpu_count()
                per = 3.7  # 默认切片长度

                cmd = [
                    python_cmd,
                    "infer/modules/train/preprocess.py",
                    dataset_path,
                    str(sr),
                    str(n_p),
                    exp_dir,
                    "False",  # noparallel
                    str(per)
                ]

                self.log_training(f"执行命令: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')

                if result.stdout:
                    self.log_training(result.stdout)
                if result.stderr:
                    self.log_training(result.stderr)

                if result.returncode == 0:
                    self.log_training("预处理完成！")
                    messagebox.showinfo("成功", "数据集预处理完成！")
                else:
                    self.log_training(f"进程退出代码: {result.returncode}")
                    messagebox.showwarning("警告", f"进程完成但有警告 (代码: {result.returncode})")

            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                self.log_training(f"错误: {error_msg}")
                messagebox.showerror("错误", f"预处理失败: {str(e)}")

        threading.Thread(target=preprocess, daemon=True).start()

    def prepare_training_config(self, exp_dir, sr, if_f0, version="v2"):
        """准备训练配置文件（filelist.txt 和 config.json）"""
        try:
            self.log_training("生成训练配置文件...")

            # 生成 filelist.txt
            gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
            feature_dir = os.path.join(exp_dir, "3_feature256" if version == "v1" else "3_feature768")

            if if_f0:
                f0_dir = os.path.join(exp_dir, "2a_f0")
                f0nsf_dir = os.path.join(exp_dir, "2b-f0nsf")
                names = (
                    set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
                    & set([name.split(".")[0] for name in os.listdir(feature_dir)])
                    & set([name.split(".")[0] for name in os.listdir(f0_dir)])
                    & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
                )
            else:
                names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
                    [name.split(".")[0] for name in os.listdir(feature_dir)]
                )

            opt = []
            for name in names:
                if if_f0:
                    opt.append(
                        "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                        % (
                            gt_wavs_dir.replace("\\", "\\\\"),
                            name,
                            feature_dir.replace("\\", "\\\\"),
                            name,
                            f0_dir.replace("\\", "\\\\"),
                            name,
                            f0nsf_dir.replace("\\", "\\\\"),
                            name,
                            0,  # speaker id
                        )
                    )
                else:
                    opt.append(
                        "%s/%s.wav|%s/%s.npy|%s"
                        % (
                            gt_wavs_dir.replace("\\", "\\\\"),
                            name,
                            feature_dir.replace("\\", "\\\\"),
                            name,
                            0,  # speaker id
                        )
                    )

            fea_dim = 256 if version == "v1" else 768
            sr_str = {32000: "32k", 40000: "40k", 48000: "48k"}.get(sr, "40k")

            # 添加 mute 数据
            if if_f0:
                for _ in range(2):
                    opt.append(
                        "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                        % (now_dir, sr_str, now_dir, fea_dim, now_dir, now_dir, 0)
                    )
            else:
                for _ in range(2):
                    opt.append(
                        "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                        % (now_dir, sr_str, now_dir, fea_dim, 0)
                    )

            shuffle(opt)
            with open(os.path.join(exp_dir, "filelist.txt"), "w") as f:
                f.write("\n".join(opt))

            self.log_training(f"已生成 filelist.txt，共 {len(opt)} 条数据")

            # 生成 config.json
            # v2 只支持 32k 和 48k，40k 使用 v1
            if sr_str == "40k":
                config_path = "v1/40k.json"
            elif version == "v1":
                config_path = f"v1/{sr_str}.json"
            else:
                config_path = f"v2/{sr_str}.json"

            config_save_path = os.path.join(exp_dir, "config.json")
            if not os.path.exists(config_save_path):
                with open(config_save_path, "w", encoding="utf-8") as f:
                    json.dump(
                        self.config.json_config[config_path],
                        f,
                        ensure_ascii=False,
                        indent=4,
                        sort_keys=True,
                    )
                    f.write("\n")
                self.log_training(f"已生成 config.json")
            else:
                self.log_training(f"config.json 已存在，跳过生成")

            return True
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            self.log_training(f"配置文件生成错误: {error_msg}")
            return False

    def extract_features(self):
        """提取特征"""
        exp_name = self.exp_name_var.get()

        if not exp_name:
            messagebox.showwarning("警告", "请输入实验名称！")
            return

        exp_dir = os.path.join(now_dir, "logs", exp_name)
        if not os.path.exists(exp_dir):
            messagebox.showwarning("警告", "实验目录不存在！请先预处理数据集！")
            return

        def extract():
            try:
                self.log_training(f"提取特征: {exp_name}")

                python_cmd = sys.executable
                version = "v2"  # 默认使用 v2

                # 提取 F0
                if self.if_f0_var.get():
                    self.log_training(f"提取音高特征 (算法: {self.train_f0_method.get()})...")
                    cmd = [
                        python_cmd,
                        "infer/modules/train/extract/extract_f0_print.py",
                        exp_dir,
                        str(os.cpu_count()),
                        self.train_f0_method.get()
                    ]
                    self.log_training(f"执行: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                    if result.stdout:
                        self.log_training(result.stdout)
                    if result.stderr:
                        self.log_training(result.stderr)

                # 提取 HuBERT 特征
                self.log_training("提取 HuBERT 特征...")
                cmd = [
                    python_cmd,
                    "infer/modules/train/extract_feature_print.py",
                    "cpu",
                    "1",
                    "0",
                    exp_dir,
                    version
                ]
                self.log_training(f"执行: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                if result.stdout:
                    self.log_training(result.stdout)
                if result.stderr:
                    self.log_training(result.stderr)

                self.log_training("特征提取完成！")
                messagebox.showinfo("成功", "特征提取完成！")

            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                self.log_training(f"错误: {error_msg}")
                messagebox.showerror("错误", f"特征提取失败: {str(e)}")

        threading.Thread(target=extract, daemon=True).start()

    def train_model(self):
        """训练模型"""
        exp_name = self.exp_name_var.get()

        if not exp_name:
            messagebox.showwarning("警告", "请输入实验名称！")
            return

        exp_dir = os.path.join(now_dir, "logs", exp_name)
        if not os.path.exists(exp_dir):
            messagebox.showwarning("警告", "实验目录不存在！请先预处理和提取特征！")
            return

        def train():
            try:
                self.log_training(f"开始训练模型: {exp_name}")

                python_cmd = sys.executable
                sr_dict = {"32k": 32000, "40k": 40000, "48k": 48000}
                sr = sr_dict[self.sr_var.get()]

                # 准备训练配置
                if not self.prepare_training_config(exp_dir, sr, self.if_f0_var.get(), "v2"):
                    messagebox.showerror("错误", "配置文件生成失败！")
                    return

                # 使用简化的训练命令
                cmd = [
                    python_cmd,
                    "infer/modules/train/train.py",
                    "-e", exp_name,
                    "-sr", str(sr),
                    "-f0", "1" if self.if_f0_var.get() else "0",
                    "-bs", str(self.batch_size.get()),
                    "-g", "0",
                    "-te", str(self.total_epoch.get()),
                    "-se", str(self.save_epoch.get()),
                    "-pg", "",
                    "-pd", "",
                    "-l", "0",
                    "-c", "0",
                    "-sw", "0",
                    "-v", "v2"
                ]

                self.log_training(f"执行命令: {' '.join(cmd)}")
                self.log_training("训练已开始，这可能需要较长时间...")
                self.log_training("警告：训练过程会在后台运行，请查看日志文件获取详细进度")

                # 使用 Popen 在后台运行
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='ignore',
                    bufsize=1
                )

                # 实时读取输出
                for line in process.stdout:
                    self.log_training(line.strip())

                process.wait()

                if process.returncode == 0:
                    self.log_training("训练完成！")
                    messagebox.showinfo("成功", "模型训练完成！")
                else:
                    self.log_training(f"训练进程退出代码: {process.returncode}")
                    messagebox.showwarning("警告", f"训练完成但有警告 (代码: {process.returncode})")

            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                self.log_training(f"错误: {error_msg}")
                messagebox.showerror("错误", f"训练失败: {str(e)}")

        threading.Thread(target=train, daemon=True).start()

    def auto_train(self):
        """一键训练"""
        if self.training_in_progress:
            messagebox.showwarning("警告", "训练正在进行中，请等待完成！")
            return

        exp_name = self.exp_name_var.get()
        dataset_path = self.dataset_path_var.get()

        if not exp_name:
            messagebox.showwarning("警告", "请输入实验名称！")
            return

        if not dataset_path or not os.path.exists(dataset_path):
            messagebox.showwarning("警告", "请选择一个有效的数据集目录！")
            return

        response = messagebox.askyesno(
            "确认",
            f"将自动执行以下步骤：\n1. 预处理数据集\n2. 提取特征\n3. 生成训练配置\n4. 训练模型\n\n实验名称: {exp_name}\n数据集: {dataset_path}\n\n确定继续？"
        )

        if not response:
            return

        def auto_train_process():
            try:
                self.training_in_progress = True
                self.log_training("="*60)
                self.log_training("开始一键训练流程")
                self.log_training("="*60)

                # 步骤 0: 清理实验目录
                sr_dict = {"32k": 32000, "40k": 40000, "48k": 48000}
                sr = sr_dict[self.sr_var.get()]
                exp_dir = os.path.join(now_dir, "logs", exp_name)

                if os.path.exists(exp_dir):
                    self.log_training("\n[步骤 0] 清理旧的训练文件...")
                    import shutil
                    try:
                        shutil.rmtree(exp_dir)
                        self.log_training("已清理旧文件")
                    except Exception as e:
                        self.log_training(f"清理失败: {e}")

                os.makedirs(exp_dir, exist_ok=True)

                # 步骤 1: 预处理
                self.log_training("\n[步骤 1/5] 预处理数据集...")

                python_cmd = sys.executable
                n_p = os.cpu_count()
                per = 3.7

                # 创建静默执行的参数（隐藏命令行窗口）
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                creation_flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0

                cmd = [python_cmd, "infer/modules/train/preprocess.py", dataset_path, str(sr), str(n_p), exp_dir, "False", str(per)]
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore',
                                       startupinfo=startupinfo, creationflags=creation_flags)
                if result.stdout:
                    self.log_training(result.stdout)
                if result.returncode != 0:
                    raise Exception(f"预处理失败 (代码: {result.returncode})")
                self.log_training("[步骤 1/5] 预处理完成！\n")

                # 步骤 2: 提取特征
                self.log_training("[步骤 2/5] 提取特征...")
                version = "v2"

                if self.if_f0_var.get():
                    self.log_training(f"提取音高特征 (算法: {self.train_f0_method.get()})...")
                    cmd = [python_cmd, "infer/modules/train/extract/extract_f0_print.py", exp_dir, str(n_p), self.train_f0_method.get()]
                    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore',
                                           startupinfo=startupinfo, creationflags=creation_flags)
                    if result.stdout:
                        self.log_training(result.stdout)
                    if result.stderr:
                        self.log_training(f"F0提取警告: {result.stderr}")

                self.log_training("提取 HuBERT 特征...")
                cmd = [python_cmd, "infer/modules/train/extract_feature_print.py", "cpu", "1", "0", exp_dir, version, "False"]
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore',
                                       startupinfo=startupinfo, creationflags=creation_flags)
                if result.stdout:
                    self.log_training(result.stdout)
                if result.stderr:
                    self.log_training(f"HuBERT提取警告: {result.stderr}")
                if result.returncode != 0:
                    raise Exception(f"HuBERT特征提取失败 (代码: {result.returncode})")
                self.log_training("[步骤 2/5] 特征提取完成！\n")

                # 步骤 3: 准备训练配置
                self.log_training("\n[步骤 3/5] 准备训练配置...")
                if not self.prepare_training_config(exp_dir, sr, self.if_f0_var.get(), "v2"):
                    raise Exception("配置文件生成失败")
                self.log_training("[步骤 3/5] 配置文件已生成！\n")

                # 步骤 4: 训练
                self.log_training("[步骤 4/5] 开始训练模型...")
                self.log_training(f"训练参数: 轮数={self.total_epoch.get()}, 批次={self.batch_size.get()}, 保存频率={self.save_epoch.get()}")

                cmd = [
                    python_cmd, "infer/modules/train/train.py",
                    "-e", exp_name,
                    "-sr", self.sr_var.get(),  # 直接使用"32k", "40k", "48k"
                    "-f0", "1" if self.if_f0_var.get() else "0",
                    "-bs", str(self.batch_size.get()),
                    "-g", "0",  # GPU编号，0表示CPU
                    "-te", str(self.total_epoch.get()),
                    "-se", str(self.save_epoch.get()),
                    "-l", "0",  # 是否从最新检查点加载
                    "-c", "0",  # 是否仅保存最新检查点
                    "-sw", "0",  # 是否保存每个权重
                    "-v", "v2"
                ]

                # 使用静默执行，隐藏命令行窗口
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                          encoding='utf-8', errors='ignore', bufsize=1,
                                          startupinfo=startupinfo, creationflags=creation_flags)

                # 实时读取输出
                while True:
                    line = process.stdout.readline()
                    if not line:
                        if process.poll() is not None:
                            break
                        continue
                    self.log_training(line.rstrip())

                # 读取stderr
                stderr_output = process.stderr.read()
                if stderr_output:
                    self.log_training(f"\n训练错误/警告:\n{stderr_output}")

                process.wait()

                if process.returncode == 0:
                    self.log_training("[步骤 4/5] 模型训练完成！\n")

                    # 步骤 5: 训练索引
                    self.log_training("[步骤 5/5] 训练索引文件...")
                    try:
                        self.train_index(exp_name, exp_dir, version)
                        self.log_training("[步骤 5/5] 索引训练完成！\n")
                    except Exception as e:
                        self.log_training(f"索引训练失败: {str(e)}")
                        self.log_training("训练可以继续使用，但没有索引文件可能影响音质")

                    self.log_training("\n" + "="*60)
                    self.log_training("一键训练完成！")
                    self.log_training("="*60)
                    messagebox.showinfo("成功", "一键训练完成！模型已保存。")
                else:
                    raise Exception(f"训练失败 (代码: {process.returncode})")

            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                self.log_training(f"\n错误: {error_msg}")
                messagebox.showerror("错误", f"一键训练失败: {str(e)}")
            finally:
                self.training_in_progress = False

        threading.Thread(target=auto_train_process, daemon=True).start()

    def log_inference(self, message):
        """在推理日志中记录信息"""
        self.inference_log.insert(tk.END, f"{message}\n")
        self.inference_log.see(tk.END)
        self.root.update_idletasks()

    def log_training(self, message):
        """在训练日志中记录信息"""
        self.training_log.insert(tk.END, f"{message}\n")
        self.training_log.see(tk.END)
        self.root.update_idletasks()

    def train_index(self, exp_name, exp_dir, version):
        """训练索引文件"""
        feature_dir = os.path.join(exp_dir, "3_feature256" if version == "v1" else "3_feature768")

        if not os.path.exists(feature_dir):
            raise Exception("特征目录不存在")

        listdir_res = list(os.listdir(feature_dir))
        if len(listdir_res) == 0:
            raise Exception("特征目录为空")

        # 加载所有特征
        self.log_training("加载特征文件...")
        npys = []
        for name in sorted(listdir_res):
            phone = np.load(os.path.join(feature_dir, name))
            npys.append(phone)

        big_npy = np.concatenate(npys, 0)
        big_npy_idx = np.arange(big_npy.shape[0])
        np.random.shuffle(big_npy_idx)
        big_npy = big_npy[big_npy_idx]

        # 如果特征太多，使用 KMeans 聚类
        if big_npy.shape[0] > 2e5:
            self.log_training(f"特征数量 {big_npy.shape[0]} 过多，使用 KMeans 聚类到 10k 中心...")
            try:
                n_cpu = os.cpu_count()
                big_npy = (
                    MiniBatchKMeans(
                        n_clusters=10000,
                        verbose=True,
                        batch_size=256 * n_cpu,
                        compute_labels=False,
                        init="random",
                    )
                    .fit(big_npy)
                    .cluster_centers_
                )
            except Exception as e:
                self.log_training(f"KMeans 聚类警告: {str(e)}")

        # 保存特征
        np.save(os.path.join(exp_dir, "total_fea.npy"), big_npy)

        # 训练索引
        n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
        self.log_training(f"特征形状: {big_npy.shape}, IVF数量: {n_ivf}")

        index = faiss.index_factory(256 if version == "v1" else 768, "IVF%s,Flat" % n_ivf)
        self.log_training("训练索引...")
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = 1
        index.train(big_npy)

        # 保存训练后的索引
        faiss.write_index(
            index,
            os.path.join(exp_dir, f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_name}_{version}.index")
        )

        # 添加向量
        self.log_training("添加向量...")
        batch_size_add = 8192
        for i in range(0, big_npy.shape[0], batch_size_add):
            index.add(big_npy[i : i + batch_size_add])

        # 保存最终索引
        index_file = os.path.join(exp_dir, f"added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_name}_{version}.index")
        faiss.write_index(index, index_file)

        self.log_training(f"索引文件已保存: {os.path.basename(index_file)}")


def main():
    root = tk.Tk()
    app = SimpleRVCGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
