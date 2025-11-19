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


class SimpleRVCGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RVC Simple GUI - Training & Inference")
        self.root.geometry("900x700")

        # 初始化配置
        self.config = Config()
        self.vc = VC(self.config)

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

        self.notebook.add(self.inference_frame, text="Inference (音频转换)")
        self.notebook.add(self.train_frame, text="Training (训练)")

        # 创建推理界面
        self.create_inference_ui()

        # 创建训练界面
        self.create_training_ui()

    def create_inference_ui(self):
        """创建推理界面"""
        frame = self.inference_frame

        # Model Selection
        ttk.Label(frame, text="Model Selection:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)

        model_frame = ttk.Frame(frame)
        model_frame.grid(row=1, column=0, columnspan=3, sticky=tk.EW, padx=10, pady=5)

        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, state="readonly", width=50)
        self.model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(model_frame, text="Refresh", command=self.refresh_models).pack(side=tk.LEFT, padx=5)
        ttk.Button(model_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT)

        # Input Audio
        ttk.Label(frame, text="Input Audio:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)

        input_frame = ttk.Frame(frame)
        input_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, padx=10, pady=5)

        self.input_path_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.input_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(input_frame, text="Browse", command=self.browse_input).pack(side=tk.LEFT, padx=5)

        # Output Audio
        ttk.Label(frame, text="Output Audio:", font=("Arial", 10, "bold")).grid(row=4, column=0, sticky=tk.W, padx=10, pady=5)

        output_frame = ttk.Frame(frame)
        output_frame.grid(row=5, column=0, columnspan=3, sticky=tk.EW, padx=10, pady=5)

        self.output_path_var = tk.StringVar(value="output.wav")
        ttk.Entry(output_frame, textvariable=self.output_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="Browse", command=self.browse_output).pack(side=tk.LEFT, padx=5)

        # Parameters
        ttk.Label(frame, text="Parameters:", font=("Arial", 10, "bold")).grid(row=6, column=0, sticky=tk.W, padx=10, pady=5)

        param_frame = ttk.Frame(frame)
        param_frame.grid(row=7, column=0, columnspan=3, sticky=tk.EW, padx=10, pady=5)

        # F0 Up Key (Pitch Shift)
        ttk.Label(param_frame, text="Pitch Shift:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.f0_up_key = tk.IntVar(value=0)
        ttk.Spinbox(param_frame, from_=-12, to=12, textvariable=self.f0_up_key, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        # F0 Method
        ttk.Label(param_frame, text="F0 Method:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.f0_method = tk.StringVar(value="harvest")
        ttk.Combobox(param_frame, textvariable=self.f0_method, values=["harvest", "crepe", "rmvpe", "pm"], state="readonly", width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # Index Rate
        ttk.Label(param_frame, text="Index Rate:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.index_rate = tk.DoubleVar(value=0.66)
        ttk.Scale(param_frame, from_=0, to=1, variable=self.index_rate, orient=tk.HORIZONTAL).grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Label(param_frame, textvariable=self.index_rate).grid(row=2, column=2, sticky=tk.W, padx=5, pady=2)

        # Filter Radius
        ttk.Label(param_frame, text="Filter Radius:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.filter_radius = tk.IntVar(value=3)
        ttk.Spinbox(param_frame, from_=0, to=7, textvariable=self.filter_radius, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)

        # Protect
        ttk.Label(param_frame, text="Protect:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.protect = tk.DoubleVar(value=0.33)
        ttk.Scale(param_frame, from_=0, to=0.5, variable=self.protect, orient=tk.HORIZONTAL).grid(row=4, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Label(param_frame, textvariable=self.protect).grid(row=4, column=2, sticky=tk.W, padx=5, pady=2)

        # Convert Button
        ttk.Button(frame, text="Convert Audio", command=self.convert_audio, style="Accent.TButton").grid(row=8, column=0, columnspan=3, pady=10)

        # Log Output
        ttk.Label(frame, text="Log Output:", font=("Arial", 10, "bold")).grid(row=9, column=0, sticky=tk.W, padx=10, pady=5)

        self.inference_log = scrolledtext.ScrolledText(frame, height=10, wrap=tk.WORD)
        self.inference_log.grid(row=10, column=0, columnspan=3, sticky=tk.NSEW, padx=10, pady=5)

        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(10, weight=1)

    def create_training_ui(self):
        """创建训练界面"""
        frame = self.train_frame

        # Experiment Name
        ttk.Label(frame, text="Experiment Name:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.exp_name_var = tk.StringVar(value="my_model")
        ttk.Entry(frame, textvariable=self.exp_name_var, width=50).grid(row=1, column=0, columnspan=3, sticky=tk.EW, padx=10, pady=5)

        # Dataset Directory
        ttk.Label(frame, text="Dataset Directory:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)

        dataset_frame = ttk.Frame(frame)
        dataset_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, padx=10, pady=5)

        self.dataset_path_var = tk.StringVar()
        ttk.Entry(dataset_frame, textvariable=self.dataset_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dataset_frame, text="Browse", command=self.browse_dataset).pack(side=tk.LEFT, padx=5)

        # Training Parameters
        ttk.Label(frame, text="Training Parameters:", font=("Arial", 10, "bold")).grid(row=4, column=0, sticky=tk.W, padx=10, pady=5)

        param_frame = ttk.Frame(frame)
        param_frame.grid(row=5, column=0, columnspan=3, sticky=tk.EW, padx=10, pady=5)

        # Sample Rate
        ttk.Label(param_frame, text="Sample Rate:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.sr_var = tk.StringVar(value="40k")
        ttk.Combobox(param_frame, textvariable=self.sr_var, values=["32k", "40k", "48k"], state="readonly", width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        # Use F0
        self.if_f0_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="Use F0 (Pitch)", variable=self.if_f0_var).grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)

        # Total Epochs
        ttk.Label(param_frame, text="Total Epochs:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.total_epoch = tk.IntVar(value=200)
        ttk.Spinbox(param_frame, from_=10, to=1000, textvariable=self.total_epoch, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # Batch Size
        ttk.Label(param_frame, text="Batch Size:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.batch_size = tk.IntVar(value=8)
        ttk.Spinbox(param_frame, from_=1, to=32, textvariable=self.batch_size, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        # Training Steps
        ttk.Label(frame, text="Training Steps:", font=("Arial", 10, "bold")).grid(row=6, column=0, sticky=tk.W, padx=10, pady=5)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=7, column=0, columnspan=3, sticky=tk.EW, padx=10, pady=5)

        ttk.Button(btn_frame, text="1. Preprocess Dataset", command=self.preprocess_dataset).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="2. Extract Features", command=self.extract_features).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="3. Train Model", command=self.train_model).pack(side=tk.LEFT, padx=5)

        # Training Log
        ttk.Label(frame, text="Training Log:", font=("Arial", 10, "bold")).grid(row=8, column=0, sticky=tk.W, padx=10, pady=5)

        self.training_log = scrolledtext.ScrolledText(frame, height=15, wrap=tk.WORD)
        self.training_log.grid(row=9, column=0, columnspan=3, sticky=tk.NSEW, padx=10, pady=5)

        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(9, weight=1)

    def refresh_models(self):
        """刷新模型列表"""
        weight_root = os.getenv("weight_root", "assets/weights")
        models = []
        if os.path.exists(weight_root):
            for name in os.listdir(weight_root):
                if name.endswith(".pth"):
                    models.append(name)
        self.model_combo['values'] = models
        if models:
            self.model_combo.current(0)

    def load_model(self):
        """加载模型"""
        model_name = self.model_var.get()
        if not model_name:
            messagebox.showwarning("Warning", "Please select a model first!")
            return

        def load():
            try:
                self.log_inference(f"Loading model: {model_name}...")
                self.vc.get_vc(model_name)
                self.log_inference(f"Model loaded successfully!")
                messagebox.showinfo("Success", f"Model {model_name} loaded!")
            except Exception as e:
                self.log_inference(f"Error loading model: {str(e)}")
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")

        threading.Thread(target=load, daemon=True).start()

    def browse_input(self):
        """选择输入音频"""
        filename = filedialog.askopenfilename(
            title="Select Input Audio",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.m4a"), ("All Files", "*.*")]
        )
        if filename:
            self.input_path_var.set(filename)

    def browse_output(self):
        """选择输出路径"""
        filename = filedialog.asksaveasfilename(
            title="Save Output Audio",
            defaultextension=".wav",
            filetypes=[("WAV Files", "*.wav"), ("All Files", "*.*")]
        )
        if filename:
            self.output_path_var.set(filename)

    def browse_dataset(self):
        """选择数据集目录"""
        dirname = filedialog.askdirectory(title="Select Dataset Directory")
        if dirname:
            self.dataset_path_var.set(dirname)

    def convert_audio(self):
        """转换音频"""
        input_path = self.input_path_var.get()
        output_path = self.output_path_var.get()

        if not input_path or not os.path.exists(input_path):
            messagebox.showwarning("Warning", "Please select a valid input audio file!")
            return

        if not output_path:
            messagebox.showwarning("Warning", "Please specify an output path!")
            return

        def convert():
            try:
                self.log_inference("Starting audio conversion...")
                self.log_inference(f"Input: {input_path}")
                self.log_inference(f"Parameters: pitch={self.f0_up_key.get()}, f0_method={self.f0_method.get()}")

                # 自动查找对应的 index 文件
                model_name = self.model_var.get()
                model_base = model_name.replace(".pth", "")
                index_root = os.getenv("index_root", "logs")
                index_file = ""

                # 尝试查找 index 文件
                model_dir = os.path.join(index_root, model_base)
                if os.path.exists(model_dir):
                    for file in os.listdir(model_dir):
                        if file.endswith(".index") and "added" in file:
                            index_file = os.path.join(model_dir, file)
                            break

                if index_file:
                    self.log_inference(f"Using index: {index_file}")
                else:
                    self.log_inference("No index file found, proceeding without index")

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

                self.log_inference(f"Inference result: {info}")

                if wav_opt is not None and len(wav_opt) == 2:
                    sr, audio = wav_opt
                    wavfile.write(output_path, sr, audio)
                    self.log_inference(f"Audio saved to: {output_path}")
                    messagebox.showinfo("Success", "Audio conversion completed!")
                else:
                    self.log_inference("Error: Invalid output format")
                    messagebox.showerror("Error", "Audio conversion failed!")

            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                self.log_inference(f"Error: {error_msg}")
                messagebox.showerror("Error", f"Conversion failed: {str(e)}")

        threading.Thread(target=convert, daemon=True).start()

    def preprocess_dataset(self):
        """预处理数据集"""
        exp_name = self.exp_name_var.get()
        dataset_path = self.dataset_path_var.get()

        if not exp_name:
            messagebox.showwarning("Warning", "Please enter an experiment name!")
            return

        if not dataset_path or not os.path.exists(dataset_path):
            messagebox.showwarning("Warning", "Please select a valid dataset directory!")
            return

        def preprocess():
            try:
                self.log_training(f"Preprocessing dataset: {dataset_path}")
                self.log_training(f"Experiment: {exp_name}")

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

                self.log_training(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')

                if result.stdout:
                    self.log_training(result.stdout)
                if result.stderr:
                    self.log_training(result.stderr)

                if result.returncode == 0:
                    self.log_training("Preprocessing completed!")
                    messagebox.showinfo("Success", "Dataset preprocessing completed!")
                else:
                    self.log_training(f"Process exited with code: {result.returncode}")
                    messagebox.showwarning("Warning", f"Process completed with warnings (code: {result.returncode})")

            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                self.log_training(f"Error: {error_msg}")
                messagebox.showerror("Error", f"Preprocessing failed: {str(e)}")

        threading.Thread(target=preprocess, daemon=True).start()

    def extract_features(self):
        """提取特征"""
        exp_name = self.exp_name_var.get()

        if not exp_name:
            messagebox.showwarning("Warning", "Please enter an experiment name!")
            return

        exp_dir = os.path.join(now_dir, "logs", exp_name)
        if not os.path.exists(exp_dir):
            messagebox.showwarning("Warning", "Experiment directory not found! Please preprocess first!")
            return

        def extract():
            try:
                self.log_training(f"Extracting features for: {exp_name}")

                # 调用特征提取脚本
                python_cmd = sys.executable

                # 提取 F0
                if self.if_f0_var.get():
                    self.log_training("Extracting F0 features...")
                    cmd = f'"{python_cmd}" infer/modules/train/extract/extract_f0_print.py "{exp_dir}" 1 harvest'
                    os.system(cmd)

                # 提取 HuBERT 特征
                self.log_training("Extracting HuBERT features...")
                cmd = f'"{python_cmd}" infer/modules/train/extract_feature_print.py cpu 1 0 "{exp_dir}" v2'
                os.system(cmd)

                self.log_training("Feature extraction completed!")
                messagebox.showinfo("Success", "Feature extraction completed!")

            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                self.log_training(f"Error: {error_msg}")
                messagebox.showerror("Error", f"Feature extraction failed: {str(e)}")

        threading.Thread(target=extract, daemon=True).start()

    def train_model(self):
        """训练模型"""
        exp_name = self.exp_name_var.get()

        if not exp_name:
            messagebox.showwarning("Warning", "Please enter an experiment name!")
            return

        exp_dir = os.path.join(now_dir, "logs", exp_name)
        if not os.path.exists(exp_dir):
            messagebox.showwarning("Warning", "Experiment directory not found! Please preprocess and extract features first!")
            return

        self.log_training("Training feature is complex and requires manual setup.")
        self.log_training("Please use the original training script or Gradio web interface.")
        self.log_training(f"Experiment directory: {exp_dir}")
        messagebox.showinfo("Info", "For training, please use:\npython infer-web.py\nor check the Training tab in the web interface.")

    def log_inference(self, message):
        """在推理日志中记录信息"""
        self.inference_log.insert(tk.END, f"{message}\n")
        self.inference_log.see(tk.END)

    def log_training(self, message):
        """在训练日志中记录信息"""
        self.training_log.insert(tk.END, f"{message}\n")
        self.training_log.see(tk.END)


def main():
    root = tk.Tk()
    app = SimpleRVCGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
