import json
import os
import re
import requests
import unicodedata
from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient

app = Flask(__name__)
# Enable CORS for all routes so the browser/frontend can fetch data
CORS(app, resources={r"/*": {"origins": "*"}})

MONGO_URI = os.getenv("MONGO_URI", "mongodb://movie-db:27017/movies_db")

client = MongoClient(MONGO_URI)
db = client.get_database("movies_db")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")

# Private catalog enrichment: keep recommendation signals out of MongoDB and API responses.
MOVIE_METADATA = {
    "The Shawshank Redemption": {
        "genres": {"drama"},
        "themes": {"friendship", "hope", "redemption", "justice"},
        "keywords": {"prison", "escape", "injustice", "incarceration"},
        "explanation_context": "A prison drama where friendship and hope endure through injustice as one man's pursuit of redemption unfolds.",
    },
    "The Dark Knight": {
        "genres": {"thriller", "action", "crime"},
        "themes": {"superhero", "hero", "vigilante", "justice", "corruption"},
        "keywords": {"masked", "gotham", "batman", "criminal"},
        "explanation_context": "A superhero thriller about a masked hero confronting crime, corruption and difficult questions of justice in Gotham.",
    },
    "Superbad": {
        "genres": {"comedy"},
        "themes": {"friendship", "growing up", "coming of age"},
        "keywords": {"high school", "party", "teenager", "funny"},
        "explanation_context": "A funny coming-of-age comedy about high school friends navigating awkward situations as they grow toward adulthood.",
    },
    "Pulp Fiction": {
        "genres": {"crime", "drama"},
        "themes": {"violence", "fate", "dark humor"},
        "keywords": {"gangster", "criminal", "nonlinear", "mob"},
        "explanation_context": "An interwoven crime drama where gangsters, violence, fate and dark humor shape a series of connected stories.",
    },
    "The Hangover": {
        "genres": {"comedy"},
        "themes": {"friendship", "loyalty"},
        "keywords": {"bachelor party", "missing person", "funny", "vegas"},
        "explanation_context": "A comedy about friends whose Las Vegas bachelor party is followed by a search for a missing person.",
    },
    "Se7en": {
        "genres": {"thriller", "crime"},
        "themes": {"justice", "corruption", "sin"},
        "keywords": {"detective", "serial killer", "investigation", "murder"},
        "explanation_context": "A crime thriller following a detective investigation into murder, shaped by themes of justice, corruption and sin.",
    },
}

STOP_WORDS = {
    "a", "about", "an", "and", "for", "i", "in", "is", "movie", "of",
    "on", "something", "the", "to", "want", "watch", "wanna", "with",
}
TOKEN_ALIASES = {
    "comedies": "comedy",
    "funny": "comedy",
    "friend": "friendship",
    "friends": "friendship",
    "friendships": "friendship",
    "heroes": "hero",
    "superheroes": "superhero",
    "vigilantes": "vigilante",
}
TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def normalize_tokens(text):
    """Normalize text into useful, lower-case words for exact metadata matching."""
    normalized = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    return {
        TOKEN_ALIASES.get(token, token)
        for token in TOKEN_PATTERN.findall(normalized)
        if token not in STOP_WORDS
    }


def rank_movies(movies, preferences):
    """Return positive metadata matches, with genre > theme > keyword weights."""
    preference_tokens = normalize_tokens(preferences)
    ranked = []
    for movie in movies:
        metadata = MOVIE_METADATA.get(movie.get("title"), {})
        genre_tokens = normalize_tokens(" ".join(metadata.get("genres", set())))
        theme_tokens = normalize_tokens(" ".join(metadata.get("themes", set())))
        keyword_tokens = normalize_tokens(" ".join(metadata.get("keywords", set())))
        matched = preference_tokens & (genre_tokens | theme_tokens | keyword_tokens)
        score = (
            3 * len(preference_tokens & genre_tokens)
            + 2 * len(preference_tokens & theme_tokens)
            + len(preference_tokens & keyword_tokens)
        )
        if score:
            ranked.append((movie, score, sorted(matched)))

    # A title tie-breaker makes results stable regardless of MongoDB cursor order.
    return sorted(ranked, key=lambda item: (-item[1], item[0].get("title", "").casefold()))


INTERNAL_EXPLANATION_TERMS = {
    "algorithm", "fallback", "keyword", "keywords", "match", "matched", "matches",
    "matching", "metadata", "rank", "ranking", "score", "scoring",
}


def is_valid_model_reason(reason, title):
    """Reject generic, overly long, or implementation-facing explanations."""
    if not isinstance(reason, str):
        return False
    reason = " ".join(reason.split())
    words = TOKEN_PATTERN.findall(reason)
    if not 8 <= len(words) <= 35:
        return False

    tokens = normalize_tokens(reason)
    if tokens & INTERNAL_EXPLANATION_TERMS:
        return False

    context = MOVIE_METADATA.get(title, {}).get("explanation_context", "")
    context_tokens = normalize_tokens(context)
    # A useful reason must refer to more than a broad genre such as “comedy”.
    genres = normalize_tokens(" ".join(MOVIE_METADATA.get(title, {}).get("genres", set())))
    return len(tokens & (context_tokens - genres)) >= 2


def parse_model_reasons(response, ranked):
    """Associate valid model explanations with fixed candidate IDs or exact titles."""
    try:
        generated = response.json().get("response", "")
        parsed = json.loads(generated)
        reasons = {}
        title_by_id = {index: movie["title"] for index, (movie, _, _) in enumerate(ranked, start=1)}

        explanations = parsed.get("explanations", {})
        if isinstance(explanations, dict):
            for candidate_id, reason in explanations.items():
                if isinstance(candidate_id, str) and candidate_id.isdigit():
                    title = title_by_id.get(int(candidate_id))
                    if title and is_valid_model_reason(reason, title):
                        reasons[title] = " ".join(reason.split())[:300]

        # Accept title-based responses too, including output from older prompt formats.
        items = parsed.get("recommendations", [])
        if isinstance(items, list):
            title_lookup = {title.casefold(): title for title in title_by_id.values()}
            for item in items:
                if not isinstance(item, dict):
                    continue
                title = item.get("title")
                if isinstance(title, str):
                    title = title_lookup.get(title.strip().casefold())
                candidate_id = item.get("id")
                if not title and type(candidate_id) is int:
                    title = title_by_id.get(candidate_id)
                reason = item.get("reason")
                if title and title not in reasons and is_valid_model_reason(reason, title):
                    reasons[title] = " ".join(reason.split())[:300]
        return reasons
    except (AttributeError, TypeError, ValueError):
        return {}


def explanation_schema(candidate_count):
    """Build a JSON schema that asks Ollama for exactly one reason per candidate."""
    identifiers = [str(index) for index in range(1, candidate_count + 1)]
    return {
        "type": "object",
        "properties": {
            "explanations": {
                "type": "object",
                "properties": {identifier: {"type": "string"} for identifier in identifiers},
                "required": identifiers,
                "additionalProperties": False,
            }
        },
        "required": ["explanations"],
        "additionalProperties": False,
    }


def fallback_explanation(title):
    """Return a concise, movie-specific sentence based on curated catalog facts."""
    return MOVIE_METADATA[title]["explanation_context"]

@app.route("/api/movies", methods=["GET"])
def get_movies():
    try:
        category = request.args.get("category", "").lower().strip()
        query = {"category": category} if category else {}
        
        # EXPLICITLY exclude _id from projection: {"_id": 0}
        movies = list(db.movies.find(query, {"_id": 0}))
        
        return jsonify({
            "category": category or "all",
            "count": len(movies),
            "movies": movies
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/recommendations", methods=["POST"])
def get_recommendations():
    payload = request.get_json(silent=True) or {}
    category = str(payload.get("category", "")).strip().lower()
    preferences = str(payload.get("preferences", "")).strip()[:300]
    if category and category not in {"drama", "comedy", "thriller"}:
        return jsonify({"error": "Unsupported category."}), 400
    try:
        query = {"category": category} if category else {}
        movies = list(db.movies.find(query, {"_id": 0, "title": 1, "category": 1, "year": 1}))
        ranked = rank_movies(movies, preferences)[:3]
        if not ranked:
            return jsonify({
                "recommendations": [],
                "message": "No clear movie fit was found. Try describing a genre, theme, or kind of story.",
            }), 200

        candidates = [
            {
                "id": str(index),
                "title": movie["title"],
                "description": MOVIE_METADATA[movie["title"]]["explanation_context"],
                "relevant_details": terms,
            }
            for index, (movie, _, terms) in enumerate(ranked, start=1)
        ]
        prompt = (
            "Write one distinct, natural explanation for every movie below. "
            "Use its description and relevant details to connect it to the user's preference. "
            "Keep each explanation to one sentence of about 15–25 words. "
            "Use only facts in that movie's description; do not add plot details or repeat the same wording. "
            "Return an explanations object with every supplied ID as a key and its explanation as the value. "
            "The IDs are labels only. User preference: " + (preferences or "No preference provided") +
            " Movies: " + json.dumps(candidates, ensure_ascii=True)
        )
        response = requests.post(
            f"{OLLAMA_URL.rstrip('/')}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                  "format": explanation_schema(len(ranked)), "options": {"temperature": 0}},
            timeout=(3, 90),
        )
        response.raise_for_status()
        model_reasons = parse_model_reasons(response, ranked)
        recommendations = []
        for movie, _, _ in ranked:
            title = movie["title"]
            # Only use a model explanation for a title already selected by local ranking.
            reason = model_reasons.get(title)
            if not reason:
                reason = fallback_explanation(title)
            recommendations.append({"title": title, "reason": reason})
        return jsonify({
            "recommendations": recommendations,
            "message": ""
        }), 200
    except requests.RequestException:
        app.logger.exception("Movie recommendation request failed")
        return jsonify({"error": "Recommendations are temporarily unavailable. Check that Ollama is running and gemma3:1b is downloaded."}), 503

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
