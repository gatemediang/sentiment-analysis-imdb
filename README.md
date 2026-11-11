<p align="center">
  <img src="images/hero.png" alt="Movie review sentiment analyzer" width="100%">
</p>

# Sentiment Analysis of Movie Reviews (with Slang)

A sentiment classifier that labels movie reviews as positive or negative. It starts from a pre-trained Hugging Face model, measures how well that model copes with informal, slang-heavy and sarcastic reviews, fine-tunes it, and serves it in a Streamlit web app where anyone can type a review and see the prediction.

Live app: https://sentiment-analysis-imdb-web-app.streamlit.app\

Model on the Hub: gatemediang/distilbert-slang-sentiment
(https://huggingface.co/gatemediang/distilbert-slang-sentiment)

## Model

The base model is [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english), a 67M-parameter DistilBERT already trained for binary sentiment on SST-2. It handles standard English well but can misread slang ("this slapped", "it was mid"), sarcasm ("Oh great another superhero movie") and mixed reviews ("Rough start but it got so good").

The final model was fine-tuned on `data/slang_reviews.csv` (341 hand-labelled reviews covering Gen Z slang, internet acronyms such as GOAT, NGL and TBH, sarcasm and contrastive sentences) mixed with 800 reviews from the IMDB training set. Setup: 80/20 split of the slang data (272 train / 69 test, seed 42), IMDB reviews added to the training side only, 2 epochs, batch size 16, learning rate 2e-5, max length 256 with dynamic padding.

## Results

Both models were scored on the same 69 held-out slang reviews and on 500 random reviews from the IMDB test set, which checks that learning slang didn't damage performance on ordinary reviews.

| Accuracy | Slang test (69) | IMDB sample (500) |
|---|---|---|
| Base model | 84.1% | 89.0% |
| Fine-tuned, slang only | 85.5% | 86.6% |
| Fine-tuned, slang + IMDB mix (deployed) | 84.1% | 89.2% |

![Confusion matrices before and after fine-tuning](images/confusion_matrices.png)

Full classification reports are in the notebook.

### What the experiments showed

The first fine-tunes trained on slang reviews alone. Slang accuracy barely moved, while IMDB accuracy fell by 2 to 3 points, and the confusion matrices showed why: positive IMDB reviews misclassified as negative rose from 34 to as many as 59. The model had learned to distrust praise.

Two causes were tested. Removing sarcastic reviews whose labels can't be inferred from the text (for example "Absolutely brilliant work truly inspiring" labelled negative) did not fix the drift on its own. Mixing ordinary IMDB reviews into training did: the deployed model keeps IMDB accuracy at 89.2% with errors split evenly between the classes (25 false positives, 29 false negatives).

Fine-tuning did not improve slang accuracy beyond the base model. With 69 test reviews, one review is worth 1.4 points, so differences of one or two reviews between runs are within noise. A larger labelled slang set would be the next step.

Limitations: binary model with no neutral class (the app reports predictions under 65% confidence as uncertain), English only, small slang test set.

## The app

`app.py` loads the model from the Hugging Face Hub at startup and caches it, so no model files are stored in this repository.

Guard rails:

- Input is sanitised before validation: Unicode normalisation, control and invisible characters removed, HTML tags and links stripped, character floods ("soooooo", "!!!!!!") shortened.
- Rejected with a clear message: empty or whitespace-only text, text with no words, emoji-only text, link-only text, heavily repeated words, text over 5,000 characters.
- Accepted with a warning: reviews under three words, mostly non-Latin text (the model is English-only), reviews longer than 512 tokens (truncated, never crashing), reviews where links or emoji were removed.
- Low-confidence predictions are shown as uncertain rather than forced into a class.
- Rate limits: 10 analyses per minute per client and 120 per minute across all users, held in a shared, thread-safe limiter so refreshing the page doesn't reset them.
- Inference is serialised with a lock, since fast tokenizers aren't safe to call from several Streamlit sessions at once.
- Logs go to stdout with event types, input length and a short hash, never the review text itself.
- If the model can't be loaded, the app explains why and offers a retry instead of crashing; tracebacks are hidden from visitors (`.streamlit/config.toml`).

## Project structure

```
├── app.py                         Streamlit app
├── requirements.txt               App dependencies (used by Streamlit Community Cloud)
├── .streamlit/config.toml         Production settings
├── data/
│   └── slang_reviews.csv          Fine-tuning dataset
├── notebooks/
│   ├── imdb_sentiment_analysis_fine_tuned.ipynb   Training notebook (Colab)
│   ├── colab_cells_push_and_evaluate.py           Hub upload and before/after evaluation
│   ├── colab_cells_retrain.py                     Final training run (slang + IMDB mix)
│   └── requirements-notebook.txt                  Notebook dependencies
└── images/
    └── confusion_matrices.png
```

## Run the app locally

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/sentiment-analysis-imdb.git
cd sentiment-analysis-imdb
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

Optional settings (environment variables or `.streamlit/secrets.toml`): `MODEL_ID`, `MODEL_REVISION` (pin a Hub commit), `HF_TOKEN` (only for a private model repo), `LOG_TO_FILE=1` (rotating log file when running locally).

## Reproduce the training

Open `notebooks/imdb_sentiment_analysis_fine_tuned.ipynb` in Google Colab with a GPU runtime. Log in to the Hub (cell 1 of `colab_cells_push_and_evaluate.py`), run the cells in `colab_cells_retrain.py`, then cells 4 and 5 of `colab_cells_push_and_evaluate.py` for the evaluation and confusion matrices.
