"""Run DictaRashi on an image with pytesseract.

    pip install pytesseract pillow          # plus Tesseract 5.x itself
    python ocr_example.py page.png          # whole page: text in reading order
    python ocr_example.py line.png --line   # a single cropped text line
    python ocr_example.py page.png --tsv    # words with bounding boxes and confidences

On Windows, if tesseract is not on PATH, pass --tesseract "C:\\Program Files\\Tesseract-OCR\\tesseract.exe".
"""
import argparse
import os

import pytesseract
from PIL import Image

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
LANG = 'DictaRashi'  # models/DictaRashi.traineddata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image')
    ap.add_argument('--line', action='store_true', help='the image is one text line (--psm 13)')
    ap.add_argument('--tsv', action='store_true', help='print words with boxes instead of plain text')
    ap.add_argument('--tesseract', help='path to the tesseract executable if it is not on PATH')
    a = ap.parse_args()
    if a.tesseract:
        pytesseract.pytesseract.tesseract_cmd = a.tesseract
    # Point Tesseract at the folder holding the model. TESSDATA_PREFIX avoids --tessdata-dir quoting
    # problems with paths that contain spaces.
    os.environ['TESSDATA_PREFIX'] = MODEL_DIR
    config = '--psm 13' if a.line else '--psm 3'  # 3 = full automatic page segmentation

    img = Image.open(a.image)
    if a.tsv:
        d = pytesseract.image_to_data(img, lang=LANG, config=config, output_type=pytesseract.Output.DICT)
        for i, word in enumerate(d['text']):
            if word.strip():
                print(d['left'][i], d['top'][i], d['width'][i], d['height'][i], d['conf'][i], word, sep='\t')
    else:
        print(pytesseract.image_to_string(img, lang=LANG, config=config))


if __name__ == '__main__':
    main()
