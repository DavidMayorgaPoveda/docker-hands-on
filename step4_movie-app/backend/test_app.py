import pytest
import mongomock
from unittest.mock import patch

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
        {"title": "Se7en", "category": "thriller", "year": 1995}
    ])

    with patch("app.db", mock_db):
        from app import app
        app.config["TESTING"] = True
        with app.test_client() as test_client:
            yield test_client

def test_get_movies_filtering_and_projection(client):
    """Verify endpoint status, filtering logic, and _id exclusion."""
    # Test category filtering
    res = client.get("/api/movies?category=thriller")
    assert res.status_code == 200
    
    data = res.get_json()
    assert data["count"] == 2
    assert data["movies"][0]["title"] == "The Dark Knight"
    assert "_id" not in data["movies"][0]

    # Test all movies fallback
    res_all = client.get("/api/movies")
    assert res_all.status_code == 200
    assert res_all.get_json()["count"] == 6

def test_recommendations_only_include_catalog_titles(client):
    class FakeResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return {"response": '{"recommendations":[{"title":"Superbad","reason":"Funny"},{"title":"Invented Film","reason":"No"}]}'}

    with patch("app.requests.post", return_value=FakeResponse()) as ollama:
        response = client.post("/api/recommendations", json={"category": "comedy", "preferences": "funny"})
    assert response.status_code == 200
    assert response.get_json()["recommendations"] == [{"title": "Superbad", "reason": "Funny"}]
    assert "Superbad" in ollama.call_args.kwargs["json"]["prompt"]
    assert "The Dark Knight" not in ollama.call_args.kwargs["json"]["prompt"]
    assert "preference is the primary ranking criterion" in ollama.call_args.kwargs["json"]["prompt"]

def test_unfiltered_recommendations_compare_all_catalog_movies(client):
    class FakeResponse:
        def raise_for_status(self): pass
        def json(self): return {"response": '{"recommendations":[{"title":"Superbad","reason":"A comedy"}]}' }

    with patch("app.db.movies.find", wraps=client.application.view_functions["get_recommendations"].__globals__["db"].movies.find) as find_mock:
        with patch("app.requests.post", return_value=FakeResponse()) as ollama:
            response = client.post("/api/recommendations", json={"category": "", "preferences": "funny"})
    assert response.status_code == 200
    assert find_mock.call_args.args[0] == {}
    prompt = ollama.call_args.kwargs["json"]["prompt"]
    for title in ("The Shawshank Redemption", "The Dark Knight", "Superbad", "Pulp Fiction", "The Hangover", "Se7en"):
        assert title in prompt

def test_empty_model_recommendations_return_clear_message(client):
    class FakeResponse:
        def raise_for_status(self): pass
        def json(self): return {"response": '{"recommendations":[]}' }

    with patch("app.requests.post", return_value=FakeResponse()):
        response = client.post("/api/recommendations", json={"preferences": "something unusual"})
    assert response.status_code == 200
    assert response.get_json()["recommendations"] == []
    assert "No suitable movie" in response.get_json()["message"]

def test_recommendations_return_useful_error_when_ollama_is_down(client):
    import requests
    with patch("app.requests.post", side_effect=requests.Timeout):
        response = client.post("/api/recommendations", json={})
    assert response.status_code == 503
    assert "Ollama" in response.get_json()["error"]
