"""
Native Desktop GUI for Iris Flower Classification using Artificial Neural Network (ANN).
Built with Python's built-in Tkinter and Pillow.
Runs offline without browser or web server.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import IrisPredictor
from src.data_loader import CLASS_NAMES, SHORT_CLASSES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "web", "assets")


class IrisClassifierGUI:
    """Desktop Window Application for Iris ANN Classification."""

    def __init__(self, root):
        self.root = root
        self.root.title("Iris Flower Classifier Studio • Deep Learning ANN")
        self.root.geometry("980x720")
        self.root.minsize(900, 650)
        self.root.configure(bg="#0b1120")

        # Load Predictor
        try:
            self.predictor = IrisPredictor()
        except Exception as e:
            self.predictor = None
            print(f"Error loading model: {e}")

        # State Variables
        self.sl_var = tk.DoubleVar(value=5.1)
        self.sw_var = tk.DoubleVar(value=3.5)
        self.pl_var = tk.DoubleVar(value=1.4)
        self.pw_var = tk.DoubleVar(value=0.2)

        self.loaded_images = {}
        self.preload_flower_images()

        self.build_ui()
        self.classify()

    def preload_flower_images(self):
        """Preload and resize botanical artwork for display."""
        targets = {
            "setosa": "setosa.jpg",
            "versicolor": "versicolor.jpg",
            "virginica": "virginica.jpg"
        }
        for sp, filename in targets.items():
            path = os.path.join(ASSETS_DIR, filename)
            if os.path.exists(path):
                img = Image.open(path)
                img = img.resize((260, 260), Image.Resampling.LANCZOS)
                self.loaded_images[sp] = ImageTk.PhotoImage(img)

    def build_ui(self):
        # Header Frame
        header = tk.Frame(self.root, bg="#111827", padx=20, pady=12)
        header.pack(fill=tk.X)

        title = tk.Label(
            header,
            text="🌸 Iris Flower Classification • Artificial Neural Network (ANN)",
            font=("Outfit", 15, "bold"),
            fg="#f8fafc",
            bg="#111827"
        )
        title.pack(side=tk.LEFT)

        badge = tk.Label(
            header,
            text="100% Accuracy • 4-Layer ANN",
            font=("Outfit", 10, "bold"),
            fg="#34d399",
            bg="#064e3b",
            padx=10,
            pady=4
        )
        badge.pack(side=tk.RIGHT)

        # Main Split Content
        main_content = tk.Frame(self.root, bg="#0b1120", padx=20, pady=15)
        main_content.pack(fill=tk.BOTH, expand=True)

        # Left Column: Controls & Sliders
        left_col = tk.Frame(main_content, bg="#1e293b", padx=18, pady=18, relief=tk.FLAT)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        lbl_controls = tk.Label(
            left_col,
            text="Flower Morphometric Controls",
            font=("Outfit", 12, "bold"),
            fg="#93c5fd",
            bg="#1e293b"
        )
        lbl_controls.pack(anchor="w", pady=(0, 10))

        # Presets Bar
        preset_frame = tk.Frame(left_col, bg="#1e293b")
        preset_frame.pack(fill=tk.X, pady=(0, 15))

        btn_set = tk.Button(preset_frame, text="🌸 Setosa", font=("Outfit", 9, "bold"), bg="#1e3a8a", fg="white",
                            activebackground="#3b82f6", activeforeground="white", relief=tk.FLAT, padx=8, pady=4,
                            command=lambda: self.load_preset(5.1, 3.5, 1.4, 0.2))
        btn_set.pack(side=tk.LEFT, padx=3)

        btn_ver = tk.Button(preset_frame, text="🌺 Versicolor", font=("Outfit", 9, "bold"), bg="#065f46", fg="white",
                            activebackground="#10b981", activeforeground="white", relief=tk.FLAT, padx=8, pady=4,
                            command=lambda: self.load_preset(5.9, 3.0, 4.2, 1.5))
        btn_ver.pack(side=tk.LEFT, padx=3)

        btn_vir = tk.Button(preset_frame, text="🪻 Virginica", font=("Outfit", 9, "bold"), bg="#831843", fg="white",
                            activebackground="#ec4899", activeforeground="white", relief=tk.FLAT, padx=8, pady=4,
                            command=lambda: self.load_preset(6.5, 3.0, 5.5, 1.8))
        btn_vir.pack(side=tk.LEFT, padx=3)

        btn_rand = tk.Button(preset_frame, text="🎲 Random", font=("Outfit", 9), bg="#334155", fg="white",
                             relief=tk.FLAT, padx=8, pady=4, command=self.load_random)
        btn_rand.pack(side=tk.LEFT, padx=3)

        # Sliders
        self.create_slider(left_col, "Sepal Length (cm):", self.sl_var, 4.0, 8.0)
        self.create_slider(left_col, "Sepal Width (cm):", self.sw_var, 2.0, 4.5)
        self.create_slider(left_col, "Petal Length (cm):", self.pl_var, 1.0, 7.0)
        self.create_slider(left_col, "Petal Width (cm):", self.pw_var, 0.1, 2.5)

        # Info Box
        info_frame = tk.Frame(left_col, bg="#0f172a", padx=12, pady=10)
        info_frame.pack(fill=tk.X, pady=(15, 0))

        self.lbl_zscores = tk.Label(
            info_frame,
            text="Standardized Inputs: Z₁: -0.89 | Z₂: +1.02 | Z₃: -1.34 | Z₄: -1.31",
            font=("JetBrains Mono", 8),
            fg="#94a3b8",
            bg="#0f172a"
        )
        self.lbl_zscores.pack(anchor="w")

        # Right Column: Prediction Results & Image
        right_col = tk.Frame(main_content, bg="#1e293b", padx=18, pady=18)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        lbl_pred_title = tk.Label(
            right_col,
            text="Neural Classification Outcome",
            font=("Outfit", 12, "bold"),
            fg="#f472b6",
            bg="#1e293b"
        )
        lbl_pred_title.pack(anchor="w", pady=(0, 10))

        # Flower Image Display
        self.img_label = tk.Label(right_col, bg="#0f172a", relief=tk.RIDGE, bd=1)
        self.img_label.pack(pady=(0, 10))

        # Species Name & Confidence
        self.lbl_species = tk.Label(
            right_col,
            text="Iris setosa",
            font=("Outfit", 18, "bold"),
            fg="#38bdf8",
            bg="#1e293b"
        )
        self.lbl_species.pack()

        self.lbl_confidence = tk.Label(
            right_col,
            text="Confidence: 99.9%",
            font=("JetBrains Mono", 12, "bold"),
            fg="#34d399",
            bg="#1e293b"
        )
        self.lbl_confidence.pack(pady=(2, 10))

        # Probabilities
        self.prob_labels = {}
        for sp, color in [("Iris setosa", "#38bdf8"), ("Iris versicolor", "#10b981"), ("Iris virginica", "#f43f5e")]:
            row = tk.Frame(right_col, bg="#0f172a", padx=10, pady=5)
            row.pack(fill=tk.X, pady=3)

            sp_name = tk.Label(row, text=sp, font=("Outfit", 9, "bold"), fg="white", bg="#0f172a", width=14, anchor="w")
            sp_name.pack(side=tk.LEFT)

            p_val = tk.Label(row, text="0.0%", font=("JetBrains Mono", 9, "bold"), fg=color, bg="#0f172a", width=8, anchor="e")
            p_val.pack(side=tk.RIGHT)

            self.prob_labels[sp] = p_val

    def create_slider(self, parent, label_text, var, from_val, to_val):
        frame = tk.Frame(parent, bg="#1e293b")
        frame.pack(fill=tk.X, pady=6)

        header_sub = tk.Frame(frame, bg="#1e293b")
        header_sub.pack(fill=tk.X)

        lbl = tk.Label(header_sub, text=label_text, font=("Outfit", 9, "bold"), fg="#f1f5f9", bg="#1e293b")
        lbl.pack(side=tk.LEFT)

        val_lbl = tk.Label(header_sub, text=f"{var.get():.2f}", font=("JetBrains Mono", 9, "bold"), fg="#38bdf8", bg="#1e293b")
        val_lbl.pack(side=tk.RIGHT)

        def on_slide(val):
            val_lbl.config(text=f"{float(val):.2f}")
            self.classify()

        scale = tk.Scale(
            frame,
            from_=from_val,
            to=to_val,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=var,
            showvalue=False,
            bg="#334155",
            troughcolor="#0f172a",
            activebackground="#818cf8",
            highlightthickness=0,
            command=on_slide
        )
        scale.pack(fill=tk.X, pady=(2, 0))

    def load_preset(self, sl, sw, pl, pw):
        self.sl_var.set(sl)
        self.sw_var.set(sw)
        self.pl_var.set(pl)
        self.pw_var.set(pw)
        self.classify()

    def load_random(self):
        self.sl_var.set(round(float(np.random.uniform(4.3, 7.9)), 1))
        self.sw_var.set(round(float(np.random.uniform(2.0, 4.4)), 1))
        self.pl_var.set(round(float(np.random.uniform(1.0, 6.9)), 1))
        self.pw_var.set(round(float(np.random.uniform(0.1, 2.5)), 1))
        self.classify()

    def classify(self):
        if self.predictor is None:
            return

        sl = self.sl_var.get()
        sw = self.sw_var.get()
        pl = self.pl_var.get()
        pw = self.pw_var.get()

        res = self.predictor.predict(sl, sw, pl, pw)
        species = res["predicted_species"]
        short_sp = res["short_name"]
        conf = res["confidence"] * 100

        self.lbl_species.config(text=species)
        self.lbl_confidence.config(text=f"Confidence: {conf:.1f}%")

        if short_sp == "setosa":
            self.lbl_species.config(fg="#38bdf8")
        elif short_sp == "versicolor":
            self.lbl_species.config(fg="#10b981")
        else:
            self.lbl_species.config(fg="#f43f5e")

        # Update Image
        if short_sp in self.loaded_images:
            self.img_label.config(image=self.loaded_images[short_sp])

        # Update Probabilities
        for sp_name, prob in res["probabilities"].items():
            if sp_name in self.prob_labels:
                self.prob_labels[sp_name].config(text=f"{prob * 100:.1f}%")

        # Update Z-scores
        z1 = (sl - self.predictor.mean[0]) / self.predictor.std[0]
        z2 = (sw - self.predictor.mean[1]) / self.predictor.std[1]
        z3 = (pl - self.predictor.mean[2]) / self.predictor.std[2]
        z4 = (pw - self.predictor.mean[3]) / self.predictor.std[3]
        self.lbl_zscores.config(text=f"Z-Scores: Z₁:{z1:+.2f} | Z₂:{z2:+.2f} | Z₃:{z3:+.2f} | Z₄:{z4:+.2f}")


def main():
    root = tk.Tk()
    app = IrisClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
