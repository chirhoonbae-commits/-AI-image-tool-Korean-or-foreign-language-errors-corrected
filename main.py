import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
import easyocr
import threading
import os

class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Text Replacer")
        self.root.geometry("1200x800")

        # Initialize OCR Reader (Korean and English)
        # This might take a moment to load on first run
        self.reader = None 
        
        self.image_path = None
        self.original_image = None  # PIL Image
        self.display_image = None   # PIL Image for display (with boxes)
        self.tk_image = None
        self.ocr_results = []       # List of (bbox, text, prob)
        self.scale = 1.0
        
        self.setup_ui()
        
        # Initialize OCR in a separate thread to not freeze UI
        threading.Thread(target=self.init_ocr, daemon=True).start()

    def init_ocr(self):
        print("Initializing EasyOCR...")
        self.status_label.config(text="OCR 엔진 초기화 중... (잠시만 기다려주세요)")
        try:
            self.reader = easyocr.Reader(['ko', 'en'], gpu=False) # Set gpu=True if CUDA is available
            self.status_label.config(text="OCR 엔진 준비 완료")
            self.process_btn.config(state=tk.NORMAL)
        except Exception as e:
            self.status_label.config(text=f"OCR 초기화 실패: {str(e)}")
            print(f"Error initializing OCR: {e}")

    def setup_ui(self):
        # Top Control Panel
        control_frame = tk.Frame(self.root, padx=10, pady=10)
        control_frame.pack(fill=tk.X)

        tk.Button(control_frame, text="이미지 열기", command=self.open_image).pack(side=tk.LEFT, padx=5)
        self.process_btn = tk.Button(control_frame, text="OCR 분석 실행", command=self.run_ocr, state=tk.DISABLED)
        self.process_btn.pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="이미지 저장", command=self.save_image).pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(control_frame, text="준비", fg="blue")
        self.status_label.pack(side=tk.LEFT, padx=20)

        # Main Content Area
        content_frame = tk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas for Image
        self.canvas_frame = tk.Frame(content_frame, bg="gray")
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbars
        h_bar = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_bar.pack(side=tk.BOTTOM, fill=tk.X)
        v_bar = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_bar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas.configure(xscrollcommand=h_bar.set, yscrollcommand=v_bar.set)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Right Side Panel for Editing
        self.edit_frame = tk.Frame(content_frame, width=300, bg="#f0f0f0", padx=10, pady=10)
        self.edit_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.edit_frame.pack_propagate(False)

        tk.Label(self.edit_frame, text="텍스트 편집", font=("Arial", 12, "bold")).pack(pady=10)
        
        tk.Label(self.edit_frame, text="감지된 텍스트:").pack(anchor="w")
        self.detected_text_var = tk.StringVar()
        tk.Entry(self.edit_frame, textvariable=self.detected_text_var, state="readonly").pack(fill=tk.X, pady=5)
        
        tk.Label(self.edit_frame, text="변경할 텍스트:").pack(anchor="w")
        self.new_text_entry = tk.Entry(self.edit_frame)
        self.new_text_entry.pack(fill=tk.X, pady=5)
        
        tk.Button(self.edit_frame, text="적용 (Stretch)", command=self.apply_text_change, bg="#4CAF50", fg="white").pack(fill=tk.X, pady=20)
        
        self.selected_bbox_index = None

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.image_path = file_path
            self.original_image = Image.open(file_path).convert("RGB")
            self.display_image = self.original_image.copy()
            self.ocr_results = []
            self.selected_bbox_index = None
            self.show_image()
            self.status_label.config(text="이미지 로드됨. OCR 분석을 실행하세요.")

    def show_image(self):
        if self.display_image:
            self.tk_image = ImageTk.PhotoImage(self.display_image)
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            self.canvas.config(scrollregion=(0, 0, self.display_image.width, self.display_image.height))

    def run_ocr(self):
        if not self.original_image:
            return
        
        self.status_label.config(text="OCR 분석 중... 잠시만 기다려주세요.")
        self.root.update()
        
        threading.Thread(target=self._ocr_thread).start()

    def _ocr_thread(self):
        try:
            # Convert PIL to bytes for EasyOCR or path
            # EasyOCR supports file path directly
            results = self.reader.readtext(self.image_path)
            # results format: ([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], text, confidence)
            
            self.ocr_results = results
            self.draw_boxes()
            self.status_label.config(text=f"분석 완료: {len(results)}개의 텍스트 영역 감지됨")
        except Exception as e:
            self.status_label.config(text=f"OCR 오류: {str(e)}")
            print(e)

    def draw_boxes(self):
        # Draw boxes on a copy of the image
        draw_img = self.original_image.copy()
        draw = ImageDraw.Draw(draw_img)
        
        for i, (bbox, text, conf) in enumerate(self.ocr_results):
            # bbox is list of 4 points
            p0, p1, p2, p3 = bbox
            # Draw rectangle
            draw.line([tuple(p0), tuple(p1), tuple(p2), tuple(p3), tuple(p0)], fill="red", width=2)
        
        self.display_image = draw_img
        self.show_image()

    def on_canvas_click(self, event):
        # Translate canvas coordinates to image coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Find clicked box
        clicked_index = -1
        min_dist = float('inf')
        
        for i, (bbox, text, conf) in enumerate(self.ocr_results):
            p0, p1, p2, p3 = bbox
            x_min = min(p0[0], p3[0])
            x_max = max(p1[0], p2[0])
            y_min = min(p0[1], p1[1])
            y_max = max(p2[1], p3[1])
            
            if x_min <= canvas_x <= x_max and y_min <= canvas_y <= y_max:
                clicked_index = i
                break
        
        if clicked_index != -1:
            self.selected_bbox_index = clicked_index
            bbox, text, conf = self.ocr_results[clicked_index]
            self.detected_text_var.set(text)
            self.new_text_entry.delete(0, tk.END)
            self.new_text_entry.insert(0, text)
            
            # Highlight selected box
            self.highlight_box(clicked_index)

    def highlight_box(self, index):
        draw_img = self.original_image.copy()
        draw = ImageDraw.Draw(draw_img)
        
        for i, (bbox, text, conf) in enumerate(self.ocr_results):
            p0, p1, p2, p3 = bbox
            color = "blue" if i == index else "red"
            width = 4 if i == index else 2
            draw.line([tuple(p0), tuple(p1), tuple(p2), tuple(p3), tuple(p0)], fill=color, width=width)
            
        self.display_image = draw_img
        self.show_image()

    def apply_text_change(self):
        if self.selected_bbox_index is None:
            messagebox.showwarning("경고", "텍스트 영역을 먼저 선택해주세요.")
            return
            
        new_text = self.new_text_entry.get()
        if not new_text:
            return

        # Get bbox info
        bbox, old_text, conf = self.ocr_results[self.selected_bbox_index]
        p0, p1, p2, p3 = bbox
        
        # Calculate width and height of the box
        width = int(max(p1[0], p2[0]) - min(p0[0], p3[0]))
        height = int(max(p3[1], p2[1]) - min(p0[1], p1[1]))
        
        x_min = int(min(p0[0], p3[0]))
        y_min = int(min(p0[1], p1[1]))

        # 1. Inpaint (remove old text) - Simple approach: fill with average color or surrounding
        # For simplicity in this prototype, we'll crop the region, calculate average color, and fill rectangle
        # A better approach uses cv2.inpaint but requires a mask.
        
        # Let's try to pick a background color from the corner of the bbox
        # Convert part of image to numpy for color picking
        img_np = np.array(self.original_image)
        # Sample a few pixels around the border to guess background color
        # This is a naive heuristic
        bg_color = img_np[y_min, x_min] # Top-left pixel color
        bg_color_tuple = tuple(map(int, bg_color))

        # Draw filled rectangle over the old text on the ORIGINAL image
        draw = ImageDraw.Draw(self.original_image)
        draw.polygon([tuple(p0), tuple(p1), tuple(p2), tuple(p3)], fill=bg_color_tuple)

        # 2. Create new text image
        # Create a temporary large image to draw text clearly
        temp_font_size = 100
        try:
            # Try to use a Korean compatible font if available, else default
            font = ImageFont.truetype("malgun.ttf", temp_font_size) # Windows default Korean font
        except:
            font = ImageFont.load_default()
            
        # Measure text size
        dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        text_bbox = dummy_draw.textbbox((0, 0), new_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        
        # Create image with text
        text_img = Image.new('RGBA', (text_w, text_h + 20), (0, 0, 0, 0)) # Transparent bg
        text_draw = ImageDraw.Draw(text_img)
        
        # Determine text color - contrasting to bg? Or just black/white?
        # For now, let's default to Black or White based on bg brightness
        brightness = (bg_color_tuple[0] * 299 + bg_color_tuple[1] * 587 + bg_color_tuple[2] * 114) / 1000
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        
        text_draw.text((0, 0), new_text, font=font, fill=text_color)
        
        # 3. Resize (Stretch) text image to fit the bbox
        resized_text_img = text_img.resize((width, height), Image.Resampling.LANCZOS)
        
        # 4. Paste onto original image
        self.original_image.paste(resized_text_img, (x_min, y_min), resized_text_img)
        
        # Update the list with new text so we can edit it again if needed (optional, but good for consistency)
        self.ocr_results[self.selected_bbox_index] = (bbox, new_text, 1.0)
        
        # Refresh display
        self.highlight_box(self.selected_bbox_index)
        self.status_label.config(text="텍스트 변경 완료")

    def save_image(self):
        if not self.original_image:
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if file_path:
            self.original_image.save(file_path)
            messagebox.showinfo("저장 완료", f"이미지가 저장되었습니다: {file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()
