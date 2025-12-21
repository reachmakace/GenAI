
from PyPDF2 import PdfReader
from typing import List, Dict
import os
from pdf2image import convert_from_path

def extract_slides_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extracts text from each page (slide) in a PDF and returns a list of dicts with metadata.
    """
    reader = PdfReader(pdf_path)
    slides = []
    # Create output dir for images
    img_dir = os.path.join(os.path.dirname(pdf_path), "slide_images")
    os.makedirs(img_dir, exist_ok=True)
    # Only generate images for slides that don't already have them
    images = None
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        cleaned_text = clean_slide_text(text)
        img_path = os.path.join(img_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_slide_{i+1}.png")
        if not os.path.exists(img_path):
            if images is None:
                from pdf2image import convert_from_path
                images = convert_from_path(pdf_path)
            images[i].save(img_path, "PNG")
        slides.append({
            'slide_number': i + 1,
            'text': cleaned_text,
            'pdf_file': os.path.basename(pdf_path),
            'image_path': img_path
        })
    return slides

def clean_slide_text(text: str) -> str:
    """
    Simple cleaning: remove extra whitespace and normalize text.
    """
    # Remove leading/trailing whitespace and replace multiple spaces/newlines with a single space
    import re
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

if __name__ == "__main__":
    # Example usage: verify image extraction
    slides = extract_slides_from_pdf("nvidia.pdf")
    for slide in slides:
        print(f"Slide {slide['slide_number']}: {slide['text'][:100]}...")
        print(f"  Image path: {slide['image_path']}")
