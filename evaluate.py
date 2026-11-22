"""Score a Tesseract model on the test set: crop every line box, OCR it as a raw text line (--psm 13),
and report CER / WER against the ground truth. (--psm 7 returns nothing on ~2% of these crops.)

Default metric is the paper's "letters" normalization: Unicode NFC, niqqud and cantillation removed, and
everything other than Hebrew letters, digits and spaces dropped (punctuation ignored). --norm safe keeps
punctuation. Default line set is the 1,120 body lines (running headers and page numbers excluded).

Usage:
    python evaluate.py                                   # models/DictaRashi.traineddata
    python evaluate.py --model path/to/other.traineddata --all-lines --out hyp.jsonl
Requires: tesseract (5.x) on PATH or --tesseract, plus `pip install pytesseract pillow`.
"""
import argparse
import json
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import pytesseract
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
NIKUD = re.compile('[֑-ׇֽֿׁׂׅׄ]')  # points + cantillation, not maqaf/sof pasuq
CONTROL = re.compile('[​-‏‪-‮﻿]')  # zero-width + bidi marks
QUOTES = str.maketrans({'״': '"', '“': '"', '”': '"', '׳': "'", '‘': "'", '’': "'",
                        '־': '-', '–': '-', '—': '-', '‐': '-',
                        '[': '(', ']': ')', '{': '(', '}': ')'})


def norm(s, mode):
    """safe: NFC, quotes/geresh -> ' ", dashes/maqaf -> -, brackets -> ( ), no niqqud, no space before .,:;!?)
    letters: safe, then only Hebrew letters, digits and spaces kept."""
    s = unicodedata.normalize('NFC', s).replace('\t', ' ')
    s = CONTROL.sub('', NIKUD.sub('', s)).translate(QUOTES).replace("''", '"')
    s = re.sub(r'\(\s+', '(', re.sub(r'\s+([.,:;!?)])', r'\1', ' '.join(s.split())))
    if mode == 'letters':
        s = re.sub('[^א-ת0-9 ]', ' ', s)
    return ' '.join(s.split())


def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(HERE, 'models', 'DictaRashi.traineddata'))
    ap.add_argument('--tesseract', help='path to the tesseract executable if it is not on PATH')
    ap.add_argument('--norm', choices=('letters', 'safe'), default='letters')
    ap.add_argument('--all-lines', action='store_true', help='also score running headers and page numbers')
    ap.add_argument('--pad', type=int, default=0, help='pixels added around each line box')
    ap.add_argument('--jobs', type=int, default=os.cpu_count())
    ap.add_argument('--out', help='write {image, line, text, hyp} per line to this .jsonl')
    a = ap.parse_args()
    if a.tesseract:
        pytesseract.pytesseract.tesseract_cmd = a.tesseract
    model_dir, lang = os.path.split(os.path.abspath(a.model))
    lang = lang.removesuffix('.traineddata')
    os.environ['TESSDATA_PREFIX'] = model_dir  # not --tessdata-dir: pytesseract mangles quoted paths on Windows
    config = '--psm 13'

    with open(os.path.join(HERE, 'test-set', 'lines.json'), encoding='utf-8') as f:
        lines = [l for l in json.load(f) if a.all_lines or l['body']]
    pages = {}

    def ocr(l):
        x, y, w, h = l['bbox']
        img = pages[l['image']]
        crop = img.crop((max(x - a.pad, 0), max(y - a.pad, 0),
                         min(x + w + a.pad, img.width), min(y + h + a.pad, img.height)))
        return pytesseract.image_to_string(crop, lang=lang, config=config).strip()

    for l in lines:  # load pages up front: PIL images are not safe to open concurrently
        if l['image'] not in pages:
            pages[l['image']] = Image.open(os.path.join(HERE, 'test-set', 'images', l['image']))
            pages[l['image']].load()
    with ThreadPoolExecutor(a.jobs) as ex:
        hyps = list(ex.map(ocr, lines))

    ce = cn = we = wn = 0
    for l, h in zip(lines, hyps):
        g, r = norm(l['text'], a.norm), norm(h, a.norm)
        ce += lev(g, r); cn += len(g)
        we += lev(g.split(), r.split()); wn += len(g.split())
    print(f'{os.path.basename(a.model)}: {len(lines)} lines, {a.norm} norm: '
          f'CER {100 * ce / cn:.2f}%  WER {100 * we / wn:.2f}%')
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            for l, h in zip(lines, hyps):
                f.write(json.dumps({'image': l['image'], 'line': l['line'], 'text': l['text'], 'hyp': h},
                                   ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
