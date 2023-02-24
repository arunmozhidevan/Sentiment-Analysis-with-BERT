# Sentiment Analysis with BERT

## Overview

Sentiment analysis is a natural language processing technique that involves analyzing text data to determine the sentiment or emotion expressed within the text. BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model developed by Google that is capable of understanding the context of words in a sentence, and is often used as a basis for sentiment analysis tasks.

## Requirements

* beautifulsoup4
* transformers

## Installation

Install the necessary libraries

```bash
pip install -r requirements.txt
```

## Explanation

### NLP model

The model used is a [bert-base-multilingual-uncased](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) model from Hugging Face which is finetuned for sentiment analysis on product reviews in six languages: English, Dutch, German, French, Spanish and Italian. It predicts the sentiment of the review as a number of stars (between 1 and 5).

```py
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
```

### Web scrapping

For reviews we used a web scrap function to scrap the data using beautifulsoup4 from yelp web site.

```py
def getReviews(link):
    webpage = requests.get(link).text
    soup = BeautifulSoup(webpage, 'html.parser')
    pages = soup.find_all('p', 'comment__09f24__gu0rG css-qgunke')
    return [page.text for page in pages]
```

### Sentiment score

We have **sentiment_score()** function to return the tensor using the argmax function to get the max element of the array.

```py
def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    return argmax(model(tokens).logits[0])
```
