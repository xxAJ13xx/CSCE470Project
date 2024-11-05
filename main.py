import requests
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
import re
import json

def scrape_indiedb_games(base_url, num_pages=5):
    """
    Scrape IndieDB for game data. 
    Args:
        base_url (str): The base URL for IndieDB games section.
        num_pages (int): Number of pages to scrape.
    Returns:
        List of dictionaries containing game data.
    """
    games = []
    for page in range(1, num_pages + 1):
        url = f"{base_url}?page={page}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for game_entry in soup.select('.game'):
            title = game_entry.select_one('.title').get_text(strip=True)
            description = game_entry.select_one('.summary').get_text(strip=True)
            popularity_score = game_entry.select_one('.popularity').get_text(strip=True)

            # Clean and format data
            description = re.sub(r'\s+', ' ', description)
            popularity_score = float(popularity_score.replace(',', ''))
            
            games.append({
                'title': title,
                'description': description,
                'popularity': popularity_score
            })

    return games

def preprocess_texts(games):
    """
    Preprocess game descriptions for BM25.
    Args:
        games (list): List of game dictionaries.
    Returns:
        List of tokenized descriptions.
    """
    tokenized_corpus = []
    for game in games:
        tokens = re.findall(r'\w+', game['description'].lower())
        tokenized_corpus.append(tokens)
    return tokenized_corpus

def rank_games(games, query):
    """
    Rank games based on a query using BM25.
    Args:
        games (list): List of game dictionaries.
        query (str): The query for ranking.
    Returns:
        List of games sorted by relevance.
    """
    tokenized_corpus = preprocess_texts(games)
    bm25 = BM25Okapi(tokenized_corpus)
    
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)

    ranked_games = sorted(
        zip(games, scores), key=lambda x: x[1], reverse=True
    )

    return [game[0] for game in ranked_games]

def main():
    # Base URL for IndieDB games section (adjust if necessary)
    base_url = 'https://www.indiedb.com/games'

    # Scrape game data
    games = scrape_indiedb_games(base_url)

    # Rank games based on a popularity query
    query = "popularity"  # Modify the query based on desired keywords
    ranked_games = rank_games(games, query)

    # Display top ranked games
    print("Top Ranked Games by Popularity:")
    for rank, game in enumerate(ranked_games[:10], start=1):
        print(f"{rank}. {game['title']} - Popularity Score: {game['popularity']}")

if __name__ == "__main__":
    main()