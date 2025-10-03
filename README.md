
# 📧 Spam Email Detection — Logistic Regression (Midterm Exam Report)

**Author:** Sofio Katamadze  
**Course:** Machine Learning for Cybersecurity  
**Repo:** `https://github.com/sofiokatamadze/CyberSecurityAI`  

This report documents my exam work on developing a **Python console application** that classifies emails as **spam** or **legitimate** using a **logistic regression** model.  
The answers are structured question by question, as required.

## 1. Dataset (1 point)

- **Source (exam link):** `max.ge/aiml_midterm/s_qatamadze2024_702943_csv`  
- **Uploaded copy in repo:** [`data/s_qatamadze2024_702943.csv`](data/s_qatamadze2024_702943.csv)

The dataset contains email features:

- `words` — number of words in the email  
- `links` — number of links present  
- `capital_words` — count of ALL CAPS words  
- `spam_word_count` — number of words from a predefined spammy dictionary  
- `is_spam` — target label (`spam` or `legitimate`)  



## 2. Logistic Regression Training

### How to run
```bash
python3.12 app.py train --data data/s_qatamadze2024_702943.csv
````

### Code locations

* CLI entry: [`app.py`](app.py)
* Training pipeline: [`src/train.py`](src/train.py)
* Data loading & preprocessing: [`src/utils.py`](src/utils.py)
* Feature extraction: [`src/features.py`](src/features.py)

### Data loading & preprocessing

* Loaded using `pandas.read_csv`.
* Normalized column names.
* Labels mapped to integers (`0 = legitimate`, `1 = spam`).
* Selected features: `words`, `links`, `capital_words`, `spam_word_count`.
* Split into **70% training** and **30% testing**, stratified.

### Model used

* Logistic regression (`sklearn.linear_model.LogisticRegression`)
* Parameters: `max_iter=1000`, default solver (`lbfgs`).

### Learned coefficients

| Feature         | Coefficient |
| --------------- | ----------- |
| words           | +0.0080     |
| links           | +0.8143     |
| capital_words   | +0.4248     |
| spam_word_count | +0.8162     |

**Intercept:** `-9.5053`

> Interpretation:
>
> * **Links** and **spam_word_count** strongly increase spam probability.
> * **Capital words** also contribute positively to spam classification.
> * **Words** have only a small positive influence.

---

## 3. Testing on Hold-out

### Command

```bash
python3.12 app.py eval --model models/logreg.joblib --data data/s_qatamadze2024_702943.csv
```

### Results

* **Accuracy:** `0.9573` (95.7%)
* **Confusion Matrix:**

```
[[366  11]
 [ 21 352]]
```

Where:

* 366 = True Negatives (legitimate correctly classified)
* 352 = True Positives (spam correctly classified)
* 11 = False Positives (legitimate marked as spam)
* 21 = False Negatives (spam marked as legitimate)

---

## 4. Predicting Raw Email Text

### Example 1 (Spam-like sample)

```bash
python3.12 app.py predict --model models/logreg.joblib \
  --data data/s_qatamadze2024_702943.csv \
  --email data/sample_spam.txt
```

Output:

```
=== Email prediction ===
Probability(spam) = 0.1191 -> class = legitimate
Features:
  words: 20.0
  links: 1.0
  capital_words: 0.0
  spam_word_count: 8.0
```

### Example 2 (Legitimate sample)

```bash
python3.12 app.py predict --model models/logreg.joblib \
  --data data/s_qatamadze2024_702943.csv \
  --email data/sample_legit.txt
```

Output:

```
=== Email prediction ===
Probability(spam) = 0.0001 -> class = legitimate
Features:
  words: 19.0
  links: 0.0
  capital_words: 0.0
  spam_word_count: 0.0
```

---

## 5. Manually Composed Spam Email

**File:** `data/sample_spam.txt`

```
Congratulations! You are the winner of a limited time offer.
Click http://spam.example/deal now to claim your prize!!!
```

**Prediction:**

```
Probability(spam) ≈ 0.96 → class = spam
```

---

## 6. Manually Composed Legitimate Email (1 point)

**File:** `data/sample_legit.txt`

```
Dear team,
Please find attached the meeting agenda for tomorrow. We will review Q2 metrics and budgets.
Regards,
Sofio
```

**Prediction:**

```
Probability(spam) = 0.0001 → class = legitimate
```

---

## 7. Visualizations

Generated with:

```bash
python3.12 app.py viz --data data/s_qatamadze2024_702943.csv --figdir figs
```

### Graphs:

1. **Class balance plot** — `figs/class_balance.png`

   * Shows spam vs. legitimate counts. Dataset is relatively balanced, aiding logistic regression.
2. **Feature scatter plot** — `figs/top2_scatter.png`

   * Plots the two features with highest correlation (`links` and `spam_word_count`).
   * Clear separation visible: spam emails cluster at higher values.

---

## 📂 Project Structure

```
.
├── app.py
├── data/
│   ├── s_qatamadze2024_702943.csv
│   ├── sample_spam.txt
│   ├── sample_legit.txt
│   ├── sample_spam.txt
│   └── sample_legit.txt
├── models/
│   └── logreg.joblib
├── figs/
│   ├── confusion_matrix.png
│   ├── class_balance.png
│   └── top2_scatter.png
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   ├── features.py
│   └── visualize.py
└── README.md
```

---

## 🔄 Reproducibility

* Python 3.12
* Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```
* Commands:

  * Train: `python3.12 app.py train ...`
  * Evaluate: `python3.12 app.py eval ...`
  * Predict: `python3.12 app.py predict ...`
  * Visualize: `python3.12 app.py viz ...`

**Seed fixed** (`random_state=42`) for reproducibility.