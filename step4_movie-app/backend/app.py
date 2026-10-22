import os
import requests
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
        titles = {movie["title"] for movie in movies if movie.get("title")}
        if not titles:
            return jsonify({"recommendations": [], "reason": "No catalog movies match."}), 200
        prompt = (
            "Compare every candidate movie in the catalog against the user's preference. "
            "The preference is the primary ranking criterion; do not favor the first item or catalog order. "
            "Return up to three suitable movies in best-match-first order, with a concise explanation for each. "
            "If none is suitable, return an empty recommendations array. Use exact catalog titles only. "
            'Return JSON only: {"recommendations":[{"title":"exact catalog title","reason":"short reason"}]}. '
            "Candidate catalog: " + str(movies) +
            " User preference: " + (preferences or "No preference provided") +
            " Requested category: " + (category or "all categories")
        )
        response = requests.post(
            f"{OLLAMA_URL.rstrip('/')}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                  "format": "json", "options": {"temperature": 0}},
            timeout=(3, 90),
        )
        response.raise_for_status()
        generated = response.json().get("response", "")
        import json
        parsed = json.loads(generated)
        recommendations = []
        for item in parsed.get("recommendations", [])[:3]:
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            if title in titles:
                recommendations.append({"title": title, "reason": str(item.get("reason", ""))[:300]})
        return jsonify({
            "recommendations": recommendations,
            "message": "No suitable movie matched your preference in the selected catalog." if not recommendations else ""
        }), 200
    except (requests.RequestException, ValueError, TypeError, KeyError):
        app.logger.exception("Movie recommendation request failed")
        return jsonify({"error": "Recommendations are temporarily unavailable. Check that Ollama is running and gemma3:1b is downloaded."}), 503

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
