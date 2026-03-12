"""
Collects anime data from the AniList GraphQL API.
- Handles pagination
- Respects rate limits (90 req/min)
- Caches results locally to avoid redundant calls
- Caps dataset at MAX_ANIME entries
"""

import json
import time
import os
import requests
from typing import Optional

# Configuration 
ANILIST_URL = "https://graphql.anilist.co"
CACHE_FILE  = os.path.join(os.path.dirname(__file__), "data", "anime_data.json")
MAX_ANIME   = 500          # hard cap: stay within 300-600 spec
PER_PAGE    = 50           # AniList max per page
RATE_LIMIT_DELAY = 0.7     # seconds between requests (~85 req/min, safe buffer)

# GraphQL Query to fetch the most popular anime
ANIME_QUERY = """
query ($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      total
      currentPage
      lastPage
      hasNextPage
    }
    media(type: ANIME, sort: POPULARITY_DESC) {
      id
      title {
        romaji
        english
      }
      genres
      source
      studios(isMain: true) {
        nodes {
          id
          name
        }
      }
      relations {
        edges {
          relationType
          node {
            id
            type
          }
        }
      }
    }
  }
}
"""


# API Helpers

def _post(query: str, variables: dict) -> dict:
    """Send a single GraphQL request; raises on HTTP errors."""
    response = requests.post(
        ANILIST_URL,
        json={"query": query, "variables": variables},
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    if "errors" in data:
        raise RuntimeError(f"AniList API error: {data['errors']}")
    return data


def _parse_anime(raw: dict) -> dict:
    """Flatten a raw API media object into a clean dict."""
    # Prefer English title, fall back to romaji
    title = (raw["title"].get("english") or raw["title"].get("romaji") or "Unknown")

    studios = [s["name"] for s in raw["studios"]["nodes"]]

    # Only keep anime→anime official relation types
    VALID_RELATION_TYPES = {
        "PREQUEL", "SEQUEL", "SIDE_STORY", "PARENT",
        "ALTERNATIVE", "SPIN_OFF", "ADAPTATION", "SOURCE",
        "SUMMARY", "COMPILATION",
    }
    relations = [
        {"id": edge["node"]["id"], "type": edge["relationType"]}
        for edge in raw["relations"]["edges"]
        if edge["node"]["type"] == "ANIME"
        and edge["relationType"] in VALID_RELATION_TYPES
    ]

    return {
        "id":        raw["id"],
        "title":     title,
        "genres":    raw.get("genres") or [],
        "studios":   studios,
        "source":    raw.get("source") or "UNKNOWN",
        "relations": relations,
    }


# Collection Function 

def collect_anime(max_anime: int = MAX_ANIME, force_refresh: bool = False) -> list[dict]:
    """
    Fetch up to `max_anime` popular anime from AniList.

    Parameters
    ----------
    max_anime     : int  – hard cap on number of anime collected
    force_refresh : bool – ignore local cache and re-fetch

    Returns
    -------
    List of cleaned anime dicts (also saved to CACHE_FILE).
    """
    # Return cached data if available 
    if not force_refresh and os.path.exists(CACHE_FILE):
        print(f"[data_collection] Loading cached data from {CACHE_FILE}")
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cached = json.load(f)
        print(f"[data_collection] Loaded {len(cached)} anime from cache.")
        return cached

    # Paginated fetch
    print(f"[data_collection] Fetching up to {max_anime} anime from AniList API…")
    anime_list: list[dict] = []
    page = 1

    while len(anime_list) < max_anime:
        remaining = max_anime - len(anime_list)
        per_page  = min(PER_PAGE, remaining)

        try:
            data      = _post(ANIME_QUERY, {"page": page, "perPage": per_page})
            page_info = data["data"]["Page"]["pageInfo"]
            media     = data["data"]["Page"]["media"]
        except requests.exceptions.HTTPError as e:
            # Handle 429 Too Many Requests gracefully
            if e.response is not None and e.response.status_code == 429:
                wait = int(e.response.headers.get("Retry-After", 60))
                print(f"[data_collection] Rate limited – waiting {wait}s…")
                time.sleep(wait)
                continue
            raise

        for raw in media:
            anime_list.append(_parse_anime(raw))

        print(
            f"[data_collection] Page {page}/{page_info['lastPage']} "
            f"– collected {len(anime_list)} anime so far."
        )

        if not page_info["hasNextPage"]:
            break

        page += 1
        time.sleep(RATE_LIMIT_DELAY)   # be polite to the API

    # Save to cache 
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(anime_list, f, ensure_ascii=False, indent=2)
    print(f"[data_collection] Saved {len(anime_list)} anime to {CACHE_FILE}")

    return anime_list


# Stats helper function

def dataset_stats(anime_list: list[dict]) -> dict:
    """Return basic statistics about the collected dataset."""
    all_genres  = {g for a in anime_list for g in a["genres"]}
    all_studios = {s for a in anime_list for s in a["studios"]}
    all_sources = {a["source"] for a in anime_list}
    total_rels  = sum(len(a["relations"]) for a in anime_list)

    return {
        "total_anime":    len(anime_list),
        "unique_genres":  len(all_genres),
        "unique_studios": len(all_studios),
        "unique_sources": len(all_sources),
        "total_relation_edges": total_rels,
    }

if __name__ == "__main__":
    anime = collect_anime()
    stats = dataset_stats(anime)
    print("\n ----Dataset Statistics----")
    for k, v in stats.items():
        print(f"  {k:<28} {v}")
