import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

class ImageProcessingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Обробка зображень")
        self.geometry("800x600")
        
        ttk.Label(self, text="Оберіть завдання для виконання:", font=("Arial", 14)).pack(pady=20)
        
        ttk.Button(self, text="Інвертування", command=self.open_invert_window).pack(pady=5, fill='x', padx=200)
        ttk.Button(self, text="Зміна компоненти", command=self.open_channel_mod_window).pack(pady=5, fill='x', padx=200)
        ttk.Button(self, text="RGB компоненти", command=self.open_rgb_split_window).pack(pady=5, fill='x', padx=200)
        ttk.Button(self, text="Злиття зображень", command=self.open_blend_window).pack(pady=5, fill='x', padx=200)
        ttk.Button(self, text="Матричні фільтри", command=self.open_filters_window).pack(pady=5, fill='x', padx=200)
        ttk.Button(self, text="Стеганографія", command=self.open_stego_window).pack(pady=5, fill='x', padx=200)

    # --- Допоміжні методи ---
    def load_image(self, grayscale=False):
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
        if not filepath:
            return None
        flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(filepath, flags)
        return img

    def display_image(self, cv_img, label_widget, max_width=300):
        if len(cv_img.shape) == 3:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        h, w = cv_img.shape[:2]
        ratio = max_width / float(w)
        new_h = int(h * ratio)
        resized = cv2.resize(cv_img, (max_width, new_h))
        
        img_pil = Image.fromarray(resized)
        img_tk = ImageTk.PhotoImage(img_pil)
        label_widget.configure(image=img_tk)
        label_widget.image = img_tk

    # --- Вікна завдань ---
    
    # Завдання A
    def open_invert_window(self):
        win = tk.Toplevel(self)
        win.title("Інвертування")
        
        img = self.load_image()
        if img is None: return
        
        inverted = cv2.bitwise_not(img)
        
        lbl_orig = ttk.Label(win)
        lbl_orig.pack(side="left", padx=10)
        lbl_inv = ttk.Label(win)
        lbl_inv.pack(side="right", padx=10)
        
        self.display_image(img, lbl_orig)
        self.display_image(inverted, lbl_inv)

    # Завдання B
    def open_channel_mod_window(self):
        win = tk.Toplevel(self)
        win.title("Зміна компоненти (додавання константи до Red)")
        
        img = self.load_image()
        if img is None: return
        
        b, g, r = cv2.split(img)
        
        r = cv2.add(r, 50) 
        
        modified = cv2.merge((b, g, r))
        
        lbl_orig = ttk.Label(win)
        lbl_orig.pack(side="left", padx=10)
        lbl_mod = ttk.Label(win)
        lbl_mod.pack(side="right", padx=10)
        
        self.display_image(img, lbl_orig)
        self.display_image(modified, lbl_mod)

    # Завдання C
    def open_rgb_split_window(self):
        win = tk.Toplevel(self)
        win.title("Розбивка на RGB компоненти")
        
        img = self.load_image()
        if img is None: return
        
        b, g, r = cv2.split(img)
        zeros = np.zeros_like(b)
        
        img_b = cv2.merge((b, zeros, zeros))
        img_g = cv2.merge((zeros, g, zeros))
        img_r = cv2.merge((zeros, zeros, r))
        
        lbl_r = ttk.Label(win); lbl_r.pack(side="left")
        lbl_g = ttk.Label(win); lbl_g.pack(side="left")
        lbl_b = ttk.Label(win); lbl_b.pack(side="left")
        
        self.display_image(img_r, lbl_r, max_width=250)
        self.display_image(img_g, lbl_g, max_width=250)
        self.display_image(img_b, lbl_b, max_width=250)

    # Завдання D
    def open_blend_window(self):
        win = tk.Toplevel(self)
        win.title("Злиття зображень")
        
        messagebox.showinfo("Інфо", "Оберіть перше зображення")
        img1 = self.load_image()
        messagebox.showinfo("Інфо", "Оберіть друге зображення")
        img2 = self.load_image()
        
        if img1 is None or img2 is None: return
        
        # Зображення повинні бути однакового розміру для злиття
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        lbl_res = ttk.Label(win)
        lbl_res.pack(pady=10)
        
        def update_blend(val):
            alpha = float(val)
            blended = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
            self.display_image(blended, lbl_res, max_width=400)
            
        slider = ttk.Scale(win, from_=0, to=1, orient='horizontal', command=update_blend)
        slider.set(0.5)
        slider.pack(fill='x', padx=50, pady=10)

    # Завдання E 
    def open_filters_window(self):
        win = tk.Toplevel(self)
        win.title("Матричні фільтри")
        
        img = self.load_image()
        if img is None: return
        
        lbl_img = ttk.Label(win)
        lbl_img.pack(pady=10)
        self.display_image(img, lbl_img, max_width=400)
        
        def apply_filter(f_type):
            res = img.copy()
            if f_type == "blur":
                res = cv2.GaussianBlur(img, (15, 15), 0)
            elif f_type == "sharp":
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                res = cv2.filter2D(img, -1, kernel)
            elif f_type == "median":
                res = cv2.medianBlur(img, 15)
            elif f_type == "sobel":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                res = cv2.magnitude(sobelx, sobely)
                res = cv2.convertScaleAbs(res)
                res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
                
            self.display_image(res, lbl_img, max_width=400)

        frame_btns = ttk.Frame(win)
        frame_btns.pack()
        ttk.Button(frame_btns, text="Розмиття", command=lambda: apply_filter("blur")).pack(side="left")
        ttk.Button(frame_btns, text="Чіткість", command=lambda: apply_filter("sharp")).pack(side="left")
        ttk.Button(frame_btns, text="Медіанний", command=lambda: apply_filter("median")).pack(side="left")
        ttk.Button(frame_btns, text="Собель", command=lambda: apply_filter("sobel")).pack(side="left")

    # Завдання F та G
    def open_stego_window(self):
        win = tk.Toplevel(self)
        win.title("Стеганографія (НЗБ)")
        
        messagebox.showinfo("Крок 1", "Оберіть зображення-контейнер")
        container = self.load_image()
        messagebox.showinfo("Крок 2", "Оберіть водяний знак")
        watermark = self.load_image(grayscale=True) # Завантажуємо в градаціях сірого
        
        if container is None or watermark is None:
            win.destroy()
            return

        # Бінаризація водяного знаку (значення 0 або 1)
        _, binary_wm = cv2.threshold(watermark, 127, 1, cv2.THRESH_BINARY)
        
        # Створюємо фрейми для зручного розміщення: зверху картинки, знизу налаштування
        frame_imgs = ttk.Frame(win)
        frame_imgs.pack(pady=10, fill="both", expand=True)
        
        # Підписи над картинками
        ttk.Label(frame_imgs, text="Стего-контейнер (з прихованим знаком)", font=("Arial", 12)).grid(row=0, column=0, pady=5)
        ttk.Label(frame_imgs, text="Вилучений водяний знак", font=("Arial", 12)).grid(row=0, column=1, pady=5)

        lbl_stego = ttk.Label(frame_imgs)
        lbl_stego.grid(row=1, column=0, padx=10)
        
        lbl_extracted = ttk.Label(frame_imgs)
        lbl_extracted.grid(row=1, column=1, padx=10)
        
        def process_stego(plane_str):
            try:
                plane = int(plane_str) - 1 # Переводимо 1-8 у 0-7 для зсуву
            except ValueError:
                return

            h, w = container.shape[:2]
            wm_h, wm_w = binary_wm.shape
            
            tiled_wm = np.tile(binary_wm, (h // wm_h + 1, w // wm_w + 1))[:h, :w]
            
            b_channel = container[:, :, 0]
            
            mask_val = 255 - (1 << plane)
            mask_array = np.full_like(b_channel, mask_val)
            cleared_b = cv2.bitwise_and(b_channel, mask_array)
            
            shifted_wm = np.uint8(tiled_wm << plane)
            stego_b = cv2.bitwise_or(cleared_b, shifted_wm)
            
            stego_img = container.copy()
            stego_img[:, :, 0] = stego_b
            
            self.display_image(stego_img, lbl_stego, max_width=350)
            
            extract_mask = np.full_like(stego_b, 1 << plane)
            extracted_wm = cv2.bitwise_and(stego_b, extract_mask)
            extracted_wm = (extracted_wm >> plane) * 255 # Розтягуємо 0/1 до 0/255
            
            self.display_image(extracted_wm, lbl_extracted, max_width=350)

        frame_controls = ttk.Frame(win)
        frame_controls.pack(pady=10)

        ttk.Label(frame_controls, text="Номер бітової площини (1-8):").pack(side="left", padx=5)
        plane_var = tk.StringVar(value="1")
        spinbox = ttk.Spinbox(frame_controls, from_=1, to=8, textvariable=plane_var, width=5, 
                              command=lambda: process_stego(plane_var.get()))
        spinbox.pack(side="left", padx=5)
        
        process_stego("1")

        ttk.Label(win, text="Оберіть номер бітової площини (1-8):").pack()
        plane_var = tk.StringVar(value="1")
        spinbox = ttk.Spinbox(win, from_=1, to=8, textvariable=plane_var, command=lambda: process_stego(plane_var.get()))
        spinbox.pack(pady=5)
        

if __name__ == "__main__":
    app = ImageProcessingApp()
    app.mainloop()