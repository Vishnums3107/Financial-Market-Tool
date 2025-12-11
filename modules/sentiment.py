"""
Sentiment Analysis Module

News and social media sentiment analysis:
- News API integration
- NLP sentiment scoring
- Sentiment aggregation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import warnings
import re


@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    score: float  # -1 to 1
    label: str  # Bearish, Neutral, Bullish
    confidence: float
    source: str
    headline: str = ""
    timestamp: str = ""


@dataclass
class NewsSentiment:
    """Aggregated news sentiment."""
    overall_score: float
    overall_label: str
    num_articles: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    articles: List[SentimentResult]


# Simple sentiment lexicon for basic analysis
POSITIVE_WORDS = {
    'surge', 'soar', 'jump', 'rally', 'gain', 'rise', 'climb', 'boost',
    'bullish', 'optimistic', 'positive', 'growth', 'profit', 'beat',
    'upgrade', 'outperform', 'strong', 'record', 'high', 'best',
    'breakthrough', 'success', 'win', 'exceed', 'above', 'upside'
}

NEGATIVE_WORDS = {
    'crash', 'plunge', 'drop', 'fall', 'decline', 'tumble', 'slide',
    'bearish', 'pessimistic', 'negative', 'loss', 'miss', 'cut',
    'downgrade', 'underperform', 'weak', 'low', 'worst', 'fail',
    'concern', 'risk', 'warning', 'below', 'downside', 'sell-off'
}

AMPLIFIERS = {'very', 'extremely', 'significantly', 'sharply', 'dramatically'}
NEGATIONS = {'not', 'no', 'never', 'none', 'neither', "n't"}


def simple_sentiment_score(text: str) -> float:
    """
    Calculate simple sentiment score from text.
    
    Args:
        text: Text to analyze
    
    Returns:
        Sentiment score from -1 (bearish) to 1 (bullish)
    """
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    
    positive_count = 0
    negative_count = 0
    
    for i, word in enumerate(words):
        # Check for negation in previous 3 words
        negated = any(words[max(0, i-3):i].count(neg) > 0 for neg in NEGATIONS)
        
        # Check for amplifier
        amplified = any(words[max(0, i-2):i].count(amp) > 0 for amp in AMPLIFIERS)
        weight = 1.5 if amplified else 1.0
        
        if word in POSITIVE_WORDS:
            if negated:
                negative_count += weight
            else:
                positive_count += weight
        elif word in NEGATIVE_WORDS:
            if negated:
                positive_count += weight
            else:
                negative_count += weight
    
    total = positive_count + negative_count
    if total == 0:
        return 0.0
    
    score = (positive_count - negative_count) / total
    return max(-1.0, min(1.0, score))


def score_to_label(score: float) -> str:
    """Convert sentiment score to label."""
    if score > 0.2:
        return 'Bullish'
    elif score < -0.2:
        return 'Bearish'
    return 'Neutral'


def analyze_headline(headline: str) -> SentimentResult:
    """
    Analyze a single headline.
    
    Args:
        headline: News headline text
    
    Returns:
        SentimentResult
    """
    score = simple_sentiment_score(headline)
    label = score_to_label(score)
    
    # Confidence based on magnitude
    confidence = min(1.0, abs(score) * 1.5)
    
    return SentimentResult(
        score=score,
        label=label,
        confidence=confidence,
        source='Internal',
        headline=headline,
        timestamp=datetime.now().isoformat()
    )


def try_fetch_news_newsapi(
    symbol: str,
    api_key: str,
    days: int = 7
) -> List[Dict]:
    """
    Try to fetch news from NewsAPI.
    
    Args:
        symbol: Stock symbol
        api_key: NewsAPI API key
        days: Number of days to fetch
    
    Returns:
        List of article dicts
    """
    try:
        import requests
        
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': symbol,
            'from': from_date,
            'sortBy': 'publishedAt',
            'apiKey': api_key,
            'language': 'en',
            'pageSize': 50
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get('status') == 'ok':
            return data.get('articles', [])
        return []
    
    except Exception as e:
        warnings.warn(f"NewsAPI fetch failed: {e}")
        return []


def try_fetch_news_alphavantage(
    symbol: str,
    api_key: str
) -> List[Dict]:
    """
    Try to fetch news from Alpha Vantage.
    
    Args:
        symbol: Stock symbol
        api_key: Alpha Vantage API key
    
    Returns:
        List of article dicts
    """
    try:
        import requests
        
        url = f"https://www.alphavantage.co/query"
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': symbol,
            'apikey': api_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        return data.get('feed', [])
    
    except Exception as e:
        warnings.warn(f"Alpha Vantage fetch failed: {e}")
        return []


def analyze_news_sentiment(
    symbol: str,
    headlines: List[str] = None,
    api_key: str = None,
    api_source: str = 'newsapi'
) -> NewsSentiment:
    """
    Analyze news sentiment for a symbol.
    
    Args:
        symbol: Stock symbol
        headlines: Optional list of headlines (if not fetching)
        api_key: API key for news provider
        api_source: 'newsapi' or 'alphavantage'
    
    Returns:
        NewsSentiment with aggregated results
    """
    articles = []
    
    # Fetch news if no headlines provided
    if headlines is None and api_key:
        if api_source == 'newsapi':
            raw_articles = try_fetch_news_newsapi(symbol, api_key)
            headlines = [a.get('title', '') for a in raw_articles if a.get('title')]
        elif api_source == 'alphavantage':
            raw_articles = try_fetch_news_alphavantage(symbol, api_key)
            headlines = [a.get('title', '') for a in raw_articles if a.get('title')]
    
    if not headlines:
        headlines = []
    
    # Analyze each headline
    results = []
    for headline in headlines:
        if headline:
            result = analyze_headline(headline)
            results.append(result)
    
    # Aggregate results
    if not results:
        return NewsSentiment(
            overall_score=0.0,
            overall_label='Neutral',
            num_articles=0,
            bullish_count=0,
            bearish_count=0,
            neutral_count=0,
            articles=[]
        )
    
    scores = [r.score for r in results]
    overall_score = np.mean(scores)
    
    bullish = sum(1 for r in results if r.label == 'Bullish')
    bearish = sum(1 for r in results if r.label == 'Bearish')
    neutral = sum(1 for r in results if r.label == 'Neutral')
    
    return NewsSentiment(
        overall_score=overall_score,
        overall_label=score_to_label(overall_score),
        num_articles=len(results),
        bullish_count=bullish,
        bearish_count=bearish,
        neutral_count=neutral,
        articles=results
    )


def try_reddit_sentiment(
    symbol: str,
    subreddits: List[str] = ['wallstreetbets', 'stocks', 'investing']
) -> Optional[NewsSentiment]:
    """
    Try to analyze Reddit sentiment (requires PRAW).
    
    Args:
        symbol: Stock symbol
        subreddits: Subreddits to search
    
    Returns:
        NewsSentiment or None
    """
    try:
        # This would require PRAW and Reddit API credentials
        # Placeholder for future implementation
        warnings.warn("Reddit sentiment requires PRAW setup")
        return None
    except Exception:
        return None


def format_sentiment_summary(sentiment: NewsSentiment) -> pd.DataFrame:
    """Format sentiment summary for display."""
    emoji = '🟢' if sentiment.overall_label == 'Bullish' else \
            '🔴' if sentiment.overall_label == 'Bearish' else '🟡'
    
    rows = [
        ('Overall Sentiment', f"{emoji} {sentiment.overall_label}"),
        ('Sentiment Score', f"{sentiment.overall_score:.2f}"),
        ('Articles Analyzed', sentiment.num_articles),
        ('Bullish Articles', f"🟢 {sentiment.bullish_count}"),
        ('Bearish Articles', f"🔴 {sentiment.bearish_count}"),
        ('Neutral Articles', f"🟡 {sentiment.neutral_count}")
    ]
    
    return pd.DataFrame(rows, columns=['Metric', 'Value'])


def format_articles_table(articles: List[SentimentResult]) -> pd.DataFrame:
    """Format articles for display."""
    if not articles:
        return pd.DataFrame(columns=['Headline', 'Sentiment', 'Score'])
    
    rows = []
    for a in articles[:20]:  # Limit to 20
        emoji = '🟢' if a.label == 'Bullish' else '🔴' if a.label == 'Bearish' else '🟡'
        headline = a.headline[:60] + '...' if len(a.headline) > 60 else a.headline
        rows.append({
            'Headline': headline,
            'Sentiment': f"{emoji} {a.label}",
            'Score': f"{a.score:.2f}"
        })
    
    return pd.DataFrame(rows)
