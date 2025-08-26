
````markdown
# Pedal Reconstruction with Transformer

This project focuses on the reconstruction of **sustain pedal information** in expressive piano performances using Transformer-based models. Two scenarios are supported: **Full Reconstruction** and **Partial Reconstruction**.

## Usage

### 1. Data Processing (optional)
If you want to preprocess the data from scratch:

1. Generate the tokenizer:
   ```bash
   python gen_tokenizer.py
````

2. Generate the numpy dataset:

   ```bash
   python preprocess.py
   ```

   The generated dataset will be used as the model input.

---

### 2. Train Models

* **Model 1: Full Reconstruction**

  ```bash
  python main_full_removal.py
  ```

* **Model 2: Partial Reconstruction**

  ```bash
  python main_partial_removal.py
  ```

---

### 3. Test Models

Make sure to update the `best_model_path` parameter with the path to your trained model before testing.

* Test Model 1:

  ```bash
  python test_full.py
  ```

* Test Model 2:

  ```bash
  python test_partial.py
  ```

---

## File Overview

* `main_full_removal.py`: Train the Full Reconstruction model
* `test_full.py`: Test the Full Reconstruction model
* `main_partial_removal.py`: Train the Partial Reconstruction model
* `test_partial.py`: Test the Partial Reconstruction model
* `gen_tokenizer.py`: Generate tokenizer
* `preprocess.py`: Generate numpy dataset

---

## Dependencies

Install required dependencies before running:

```bash
pip install -r requirements.txt
```

