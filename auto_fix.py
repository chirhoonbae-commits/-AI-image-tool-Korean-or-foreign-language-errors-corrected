import easyocr
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def auto_fix_image(image_path, output_path):
    print(f"Processing {image_path}...")
    
    # 1. Initialize OCR
    reader = easyocr.Reader(['ko', 'en'], gpu=False)
    
    # 2. Read Image
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: File not found at {image_path}")
        return

    # 3. Detect Text
    results = reader.readtext(image_path)
    print(f"Detected {len(results)} text regions.")

    draw = ImageDraw.Draw(img)
    
    # Load Font
    try:
        font = ImageFont.truetype("malgun.ttf", 80)
    except:
        font = ImageFont.load_default()

    found_error = False
    
    for (bbox, text, prob) in results:
        print(f"Detected: '{text}' at {bbox}")
        
        # Check for the specific error text "대한안민국" or similar
        # Removing spaces for check
        clean_text = text.replace(" ", "")
        
        if "대한안민국" in clean_text or "안민" in clean_text:
            print(f"Found error text: {text} -> Fixing to '대한민국'")
            found_error = True
            
            p0, p1, p2, p3 = bbox
            x_min = int(min(p0[0], p3[0]))
            y_min = int(min(p0[1], p1[1]))
            x_max = int(max(p1[0], p2[0]))
            y_max = int(max(p2[1], p3[1]))
            
            width = x_max - x_min
            height = y_max - y_min
            
            # 1. Inpaint (Cover old text)
            # Pick background color from top-left of bbox
            img_np = np.array(img)
            bg_color = img_np[y_min, x_min]
            bg_color_tuple = tuple(map(int, bg_color))
            
            draw.polygon([tuple(p0), tuple(p1), tuple(p2), tuple(p3)], fill=bg_color_tuple)
            
            # 2. Draw New Text "대한민국"
            new_text = "대한민국"
            
            # Create text image
            # Calculate font size to fit height? 
            # For simplicity, use fixed large font then resize
            temp_font = ImageFont.truetype("malgun.ttf", 100)
            dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
            text_bbox = dummy_draw.textbbox((0, 0), new_text, font=temp_font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            
            text_img = Image.new('RGBA', (text_w, text_h + 20), (0, 0, 0, 0))
            text_draw = ImageDraw.Draw(text_img)
            
            # Text color (White or Black based on BG)
            brightness = (bg_color_tuple[0] * 299 + bg_color_tuple[1] * 587 + bg_color_tuple[2] * 114) / 1000
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
            
            text_draw.text((0, 0), new_text, font=temp_font, fill=text_color)
            
            # Resize to fit bbox
            resized_text_img = text_img.resize((width, height), Image.Resampling.LANCZOS)
            
            # Paste
            img.paste(resized_text_img, (x_min, y_min), resized_text_img)

    if found_error:
        img.save(output_path)
        print(f"Saved corrected image to {output_path}")
    else:
        print("Target text '대한안민국' not found.")

if __name__ == "__main__":
    # Assuming the user saves the image as 'sample.png'
    auto_fix_image("sample.png", "sample_fixed.png")
