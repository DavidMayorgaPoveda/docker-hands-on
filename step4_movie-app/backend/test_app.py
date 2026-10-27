import json
from unittest.mock import patch

import mongomock
import pytest
import requests

from app import app, rank_movies


class FakeResponse:
    def __init__(self, recommendations=None, raw=None):
        self.payload = raw if raw is not None else json.dumps({"recommendations": recommendations or []})

    def raise_for_status(self):
        pass

    def json(self):
        return {"response": self.payload}


@pytest.fixture
def client():
    """Sets up an in-memory MongoDB instance and Flask test client."""
    mock_client = mongomock.MongoClient()
    mock_db = mock_client["movies_db"]
    mock_db.movies.insert_many([
        {"title": "The Shawshank Redemption", "category": "drama", "year": 1994},
        {"title": "The Dark Knight", "category": "thriller", "year": 2008},
        {"title": "Superbad", "category": "comedy", "year": 2007},
        {"title": "Pulp Fiction", "category": "drama", "year": 1994},
        {"title": "The Hangover", "category": "comedy", "year": 2009},
        {"title": "Se7en", "category": "thriller", "year": 1995},
    ])

    with patch("app.db", mock_db):
        app.config["TESTING"] = True
        with app.test_client() as test_client:
            yield test_client


def test_get_movies_filtering_and_projection(client):
    """Verify endpoint status, filtering logic, and _id exclusion."""
    response = client.get("/api/movies?category=thriller")
    assert response.status_code == 200
    data = response.get_json()
    assert data["count"] == 2
    assert data["movies"][0]["title"] == "The Dark Knight"
    assert "_id" not in data["movies"][0]

    response_all = client.get("/api/movies")
    assert response_all.status_code == 200
    assert response_all.get_json()["count"] == 6


@pytest.mark.parametrize(
    ("preferences", "expected"),
    [
        ("i wanna watch a movie about a superhero", ["The Dark Knight"]),
        ("a vigilante story", ["The Dark Knight"]),
        ("crime", ["Pulp Fiction", "Se7en", "The Dark Knight"]),
        ("a movie about friendship", ["Superbad", "The Hangover", "The Shawshank Redemption"]),
        ("comedy", ["Superbad", "The Hangover"]),
    ],
)
def test_preferences_rank_movies_from_internal_metadata(client, preferences, expected):
    with patch("app.requests.post", return_value=FakeResponse()):
        response = client.post("/api/recommendations", json={"preferences": preferences})

    assert response.status_code == 200
    data = response.get_json()
    assert [item["title"] for item in data["recommendations"]] == expected
    assert set(data) == {"recommendations", "message"}
    assert all(set(item) == {"title", "reason"} for item in data["recommendations"])
    assert all(item["reason"] for item in data["recommendations"])


def test_category_filter_is_applied_before_relevance_ranking(client):
    with patch("app.requests.post", return_value=FakeResponse()) as ollama:
        response = client.post(
            "/api/recommendations", json={"category": "comedy", "preferences": "comedy"}
        )

    assert response.status_code == 200
    titles = [item["title"] for item in response.get_json()["recommendations"]]
    assert titles == ["Superbad", "The Hangover"]
    prompt = ollama.call_args.kwargs["json"]["prompt"]
    assert "The Dark Knight" not in prompt
    assert "The Shawshank Redemption" not in prompt


def test_ranking_is_independent_of_catalog_order(client):
    movies = client.get("/api/movies").get_json()["movies"]
    forward = [item[0]["title"] for item in rank_movies(movies, "crime")]
    reversed_order = [item[0]["title"] for item in rank_movies(list(reversed(movies)), "crime")]
    assert forward == reversed_order == ["Pulp Fiction", "Se7en", "The Dark Knight"]


def test_irrelevant_ollama_suggestion_cannot_replace_top_match(client):
    response_from_model = FakeResponse([
        {"title": "The Shawshank Redemption", "reason": "A story about justice."},
        {"title": "Invented Film", "reason": "Not in the catalog."},
    ])
    with patch("app.requests.post", return_value=response_from_model) as ollama:
        response = client.post(
            "/api/recommendations", json={"preferences": "superhero"}
        )

    data = response.get_json()
    assert response.status_code == 200
    assert [item["title"] for item in data["recommendations"]] == ["The Dark Knight"]
    assert data["recommendations"][0]["reason"] == (
        "A superhero thriller about a masked hero confronting crime, corruption and difficult questions of justice in Gotham."
    )
    prompt = ollama.call_args.kwargs["json"]["prompt"]
    assert "The Dark Knight" in prompt
    assert "The Shawshank Redemption" not in prompt


def test_ollama_explains_each_selected_movie_and_receives_movie_context(client):
    generated = {
        "explanations": {
            "1": "Its high school friends bring humor and awkward growing pains to this coming-of-age comedy.",
            "2": "The friends, bachelor party and Las Vegas setting give this comedy a lively premise centered on a missing person.",
            "3": "Friendship and hope remain central in this prison drama, where injustice gives one man's pursuit of redemption its emotional weight.",
        }
    }
    with patch("app.requests.post", return_value=FakeResponse(raw=json.dumps(generated))) as ollama:
        response = client.post(
            "/api/recommendations", json={"preferences": "I wanna watch a movie about friends"}
        )

    data = response.get_json()
    assert response.status_code == 200
    assert [item["title"] for item in data["recommendations"]] == [
        "Superbad", "The Hangover", "The Shawshank Redemption"
    ]
    assert [item["reason"] for item in data["recommendations"]] == list(generated["explanations"].values())
    request_data = ollama.call_args.kwargs["json"]
    assert request_data["format"]["properties"]["explanations"]["required"] == ["1", "2", "3"]
    assert "Las Vegas bachelor party" in request_data["prompt"]
    assert "friendship and hope" in request_data["prompt"]


def test_partial_ollama_response_gets_natural_movie_specific_fallbacks(client):
    generated = {
        "explanations": {
            "1": "Features a group of friends navigating awkward situations and humor."
        }
    }
    with patch("app.requests.post", return_value=FakeResponse(raw=json.dumps(generated))):
        response = client.post(
            "/api/recommendations", json={"preferences": "friends"}
        )

    recommendations = response.get_json()["recommendations"]
    assert recommendations[0]["reason"] == generated["explanations"]["1"]
    assert recommendations[1]["reason"] == (
        "A comedy about friends whose Las Vegas bachelor party is followed by a search for a missing person."
    )
    assert recommendations[2]["reason"] == (
        "A prison drama where friendship and hope endure through injustice as one man's pursuit of redemption unfolds."
    )
    assert all("metadata" not in item["reason"].lower() for item in recommendations)


def test_invalid_or_generic_ollama_reasons_use_natural_fallbacks(client):
    generated = {
        "recommendations": [
            {"id": 1, "reason": "This matches the metadata ranking for comedy and keywords."},
            {"id": 2, "reason": "A compelling and memorable story for a fun night out."},
            {"id": 3, "reason": "An interesting and emotional movie that many viewers will enjoy."},
        ]
    }
    with patch("app.requests.post", return_value=FakeResponse(raw=json.dumps(generated))):
        response = client.post("/api/recommendations", json={"preferences": "friends"})

    recommendations = response.get_json()["recommendations"]
    expected = [
        "A funny coming-of-age comedy about high school friends navigating awkward situations as they grow toward adulthood.",
        "A comedy about friends whose Las Vegas bachelor party is followed by a search for a missing person.",
        "A prison drama where friendship and hope endure through injustice as one man's pursuit of redemption unfolds.",
    ]
    assert [item["reason"] for item in recommendations] == expected
    forbidden = ("metadata", "ranking", "scoring", "keywords", "algorithm", "fallback")
    assert not any(term in item["reason"].lower() for item in recommendations for term in forbidden)


def test_no_clear_keyword_match_returns_empty_without_arbitrary_recommendations(client):
    with patch("app.requests.post") as ollama:
        response = client.post(
            "/api/recommendations", json={"preferences": "something unusual"}
        )

    assert response.status_code == 200
    assert response.get_json() == {
        "recommendations": [],
        "message": "No clear movie fit was found. Try describing a genre, theme, or kind of story.",
    }
    assert "keyword" not in response.get_json()["message"].lower()
    ollama.assert_not_called()


def test_invalid_ollama_json_uses_local_explanation(client):
    with patch("app.requests.post", return_value=FakeResponse(raw="not JSON")):
        response = client.post("/api/recommendations", json={"preferences": "vigilante"})

    assert response.status_code == 200
    assert response.get_json()["recommendations"] == [{
        "title": "The Dark Knight",
        "reason": "A superhero thriller about a masked hero confronting crime, corruption and difficult questions of justice in Gotham.",
    }]


def test_recommendation_response_keeps_maximum_of_three(client):
    with patch("app.requests.post", return_value=FakeResponse()):
        response = client.post("/api/recommendations", json={"preferences": "crime"})

    assert response.status_code == 200
    assert len(response.get_json()["recommendations"]) <= 3


def test_recommendations_return_useful_error_when_ollama_times_out(client):
    with patch("app.requests.post", side_effect=requests.Timeout):
        response = client.post("/api/recommendations", json={"preferences": "superhero"})

    assert response.status_code == 503
    assert "Ollama" in response.get_json()["error"]
