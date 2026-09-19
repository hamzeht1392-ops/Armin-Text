# ArminText

A lightweight Persian language model built from scratch with PyTorch for next-token text generation.

## درباره پروژه

**ArminText** یک مدل کوچک تولید متن فارسی است که بدون استفاده از مدل زبانی ازپیش‌آموزش‌دیده ساخته و آموزش داده شده است.

این پروژه برای یادگیری عملی مفاهیم زیر ساخته شده است:

* Tokenization
* Next-Token Prediction
* Embeddings
* GRU
* Training & Validation
* Text Generation
* PyTorch

## ویژگی‌ها

* زبان: فارسی
* مدل: 2-Layer GRU
* Tokenizer: ByteLevel BPE
* اندازه واژگان: 12,000 توکن
* Embedding Dimension: 192
* Hidden Dimension: 256
* Sequence Length: 48
* Dropout: 0.25
* Framework: PyTorch

## Dataset

مدل با استفاده از مجموعه‌داده‌ی [TinyStories-Farsi](https://huggingface.co/datasets/taesiri/TinyStories-Farsi) آموزش داده شده است.

برای نسخه نهایی پروژه، حدود 8,000 داستان و تا حدود 600,000 کلمه از داده‌ها استفاده شده است.

## معماری

```text
Input Tokens
     ↓
Embedding
     ↓
2-Layer GRU
     ↓
Dropout
     ↓
Linear Layer
     ↓
Next Token Probabilities
```

## فایل‌های پروژه

```text
Armin-Text/
├── README.md
├── ArminText.ipynb
├── ArminText-Final.pt
├── tokenizer.json
├── train.py
├── generate.py
├── requirements.txt
└── .gitignore
```

### توضیح فایل‌ها

**ArminText.ipynb**
نسخه Notebook پروژه و مراحل آموزش مدل در Google Colab.

**ArminText-Final.pt**
وزن‌های مدل آموزش‌دیده و تنظیمات معماری.

**tokenizer.json**
Tokenizer مورد استفاده برای تبدیل متن فارسی به توکن.

**train.py**
اسکریپت آموزش مدل از ابتدا.

**generate.py**
اسکریپت تولید متن با استفاده از مدل آموزش‌دیده.

**requirements.txt**
کتابخانه‌های مورد نیاز پروژه.

## اجرا

ابتدا وابستگی‌ها را نصب کنید:

```bash
pip install -r requirements.txt
```

سپس برای تولید متن:

```bash
python generate.py "روزی روزگاری"
```

نمونه با تعداد توکن بیشتر:

```bash
python generate.py "در یک روستای کوچک" --max_new_tokens 100
```

## نکته

ArminText یک مدل کوچک آموزشی و آزمایشی است و هدف آن نمایش فرایند ساخت یک مدل تولید متن فارسی از ابتدا است.

این مدل در مقیاس مدل‌هایی مانند ChatGPT نیست و کیفیت تولید متن آن به اندازه مدل‌های زبانی بزرگ نیست.

## سازنده

**آرمین حمزه**

ساخته‌شده به عنوان یک پروژه عملی برای یادگیری ساخت و آموزش مدل‌های زبانی با Python و PyTorch.
