# DictaRashi — Tesseract OCR for Rashi script

A Tesseract OCR model for scanned Hebrew books printed in the so-called "Rashi" script, trained on over one million
manually corrected lines of text from over 100 books of Rabbinic and post-Rabbinic literature (fine-tuned from the
official `heb` model). Released with its test set, a benchmark of 1,120 manually transcribed Rashi-script lines.

| Model (1,120 test lines) | CER | WER |
|---|---|---|
| **DictaRashi** | **0.86%** | **3.38%** |
| Best of the 31 other public Hebrew Tesseract models (Boro11204 `heb_rashi7`) | 2.98% | 11.86% |
| Best of 8 frontier LLMs (Claude Fable 5.2) | 2.90% | 8.46% |

CER/WER ignore punctuation and niqqud (see [Evaluation](#evaluation)). For details see our technical report here: https://arxiv.org/abs/2611.29760

Try it online in our interactive site: https://rashiocr.dicta.org.il/

## Contents

| Path | What |
|---|---|
| `models/DictaRashi.traineddata` | Our new Tesseract model |
| `test-set/images/*.png` | Test set of scanned Rashi-script pages |
| `test-set/lines.json` | Ground-truth transcriptions of each line with bounding box coordinates |
| `ocr_example.py` | Code to run the model via Python (pytesseract) |
| `evaluate.py` | Code to run the test set  |
| `SHA256SUMS` | checksums of the model and test files |

## Usage

Install [Tesseract 5](https://tesseract-ocr.github.io/tessdoc/Installation.html). Command line:

```bash
tesseract page.png out --tessdata-dir models -l DictaRashi --psm 3
```

Or copy `models/DictaRashi.traineddata` into your `tessdata` folder and use `-l DictaRashi`.

Python ([pytesseract](https://pypi.org/project/pytesseract/)):

```bash
pip install pytesseract pillow
python ocr_example.py page.png          # full page
python ocr_example.py line.png --line   # one cropped text line
python ocr_example.py page.png --tsv    # words with boxes and confidences
```

The core of it:

```python
import os, pytesseract
from PIL import Image
os.environ["TESSDATA_PREFIX"] = "models"  # folder holding DictaRashi.traineddata
print(pytesseract.image_to_string(Image.open("page.png"), lang="DictaRashi"))
```

## Test set

14 Rashi-script pages from typical out-of-copyright Rabbinic and post-Rabbinic books, scanned anew from physical
volumes (Fujitsu ScanSnap SV600, 300 dpi). There is no duplication between the test pages and the training set.
The 1,120 scored lines hold 11,294 words and 51,749 characters. Domain experts transcribed every line; the line
bounding boxes come from Tesseract's layout segmentation (`--psm 3`) and are canonical: every system is evaluated on
these same boxes, one line at a time. `test-set/lines.json` is a flat list:

```json
[
{"image": "001_p1.png", "line": 1, "bbox": [x, y, width, height], "text": "...", "body": true},
...
]
```

- `bbox` is in pixels of the page image, top-left origin.
- `line` is the line's index on its page (gaps are lines without a transcription).
- `body: false` marks running headers and page numbers (16 lines); the reported numbers use the 1,120 `body` lines.
- Transcriptions are diplomatic: abbreviation marks and punctuation as printed.

## Evaluation

```bash
pip install pytesseract pillow
python evaluate.py                       # DictaRashi, body lines, letters normalization
python evaluate.py --model other.traineddata --out hyp.jsonl
```

Each line is cropped exactly at its `bbox` and read with `--psm 13` (raw line). Normalization (`letters`):
Unicode NFC; niqqud, cantillation and invisible bidirectional or zero-width characters removed; every character other
than Hebrew letters, digits and spaces replaced by a space; whitespace runs collapsed. CER/WER are total edit distance over total
reference length.

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Please note: although the official license is
  limited to non-commercial use, users may contact us to obtain a free license for commercial use in the context of a
  small business initiative.

## Citation

```bibtex
@misc{shmidman2026dictarashistateoftheartocrmodel,
      title={DictaRashi: A State-of-the-Art OCR Model For Rashi Script}, 
      author={Ometz Shmidman and Yonah Taurog and Aryeh Sanders and Shlomo Bluth and Moshe Koppel and Joshua Geudalia and Avi Shmidman},
      year={2026},
      eprint={2611.29760},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2611.29760}, 
}
```
