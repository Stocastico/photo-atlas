"""Direct unit tests for the hand-rolled SQL builder in :mod:`photo_atlas.search`.

The builder is otherwise only exercised transitively through ``search_photos`` /
``facets``; these pin down its pieces (normalisation, OR-within / AND-across,
joins, LIKE, date bounds, sort fallback) and guard against SQL-injection by
asserting everything goes through bound ``?`` parameters.
"""

from __future__ import annotations

from photo_atlas import search
from photo_atlas.search import _as_list, _order_by, _where


def test_as_list_normalisation():
    assert _as_list(None) == []
    assert _as_list("") == []
    assert _as_list("x") == ["x"]
    assert _as_list(["a", "", None, "b"]) == ["a", "b"]
    assert _as_list((1, 2)) == [1, 2]


def test_where_empty_filters():
    where, params = _where({})
    assert where == "" and params == []


def test_where_scalar_and_list_are_equivalent_shape():
    w1, p1 = _where({"scene": "food"})
    w2, p2 = _where({"scene": ["food", "people"]})
    assert "p.scene_type IN (?)" in w1 and p1 == ["food"]
    assert "p.scene_type IN (?, ?)" in w2 and p2 == ["food", "people"]


def test_where_person_id_builds_exists_clause():
    # The person constraint is an EXISTS subquery (one row per photo, no DISTINCT
    # needed), not a JOIN that fans the result out.
    where, params = _where({"person_id": [3, 7]})
    assert "EXISTS (SELECT 1 FROM faces ef" in where
    assert "ef.person_id IN (?, ?)" in where
    assert "JOIN" not in where
    assert params == [3, 7]


def test_where_camera_is_exact_in():
    # Camera is matched exactly (whole camera_model values from the facet), so
    # the chip count matches the result count even for substring-overlapping names.
    where, params = _where({"camera": ["iPhone 15", "Pixel 8"]})
    assert "p.camera_model IN (?, ?)" in where
    assert params == ["iPhone 15", "Pixel 8"]


def test_where_q_escapes_wildcards_and_sets_escape_clause():
    where, params = _where({"q": "a_b%c"})
    assert "ESCAPE '\\'" in where
    assert params == ["%a\\_b\\%c%"] * 7


def test_where_date_bounds_and_has_faces():
    where, params = _where(
        {"date_from": "2012-01-01", "date_to": "2012-12-31", "has_faces": True}
    )
    assert "substr(p.taken_at, 1, 10) >= ?" in where
    assert "substr(p.taken_at, 1, 10) <= ?" in where
    assert "p.face_count > 0" in where
    assert params == ["2012-01-01", "2012-12-31"]


def test_where_q_spans_all_text_columns():
    where, params = _where({"q": "barca"})
    for col in ("filename", "place_city", "place_country", "place_label",
                "folder_place", "camera_make", "camera_model"):
        assert f"p.{col} LIKE ?" in where
    assert params == ["%barca%"] * 7


def test_where_combines_across_facets_with_and():
    where, _params = _where({"scene": "food", "country": "Italy"})
    assert " AND " in where


def test_order_by_falls_back_to_newest():
    assert _order_by(None) == search.SORTS["newest"]
    assert _order_by("bogus") == search.SORTS["newest"]
    assert _order_by("oldest") == search.SORTS["oldest"]
