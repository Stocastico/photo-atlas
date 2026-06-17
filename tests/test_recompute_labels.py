"""`index --recompute` must not destroy manual face naming.

A re-index re-detects faces and `db.replace_faces` blanket-deletes + re-inserts
with only the *auto* k-NN identities, which would wipe every human `assign_face`
(confidence 1.0). The indexer now carries a prior human label onto the newly
detected face whose bounding box overlaps it (IoU >= threshold), so a recompute
refreshes detection/embeddings without losing the naming work.
"""

from __future__ import annotations

from photo_atlas import db
from photo_atlas.indexer import _bbox_iou, _carry_human_labels, _match_human_labels


# -- pure IoU geometry ------------------------------------------------------
def test_bbox_iou_identity_and_disjoint():
    assert _bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _bbox_iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_bbox_iou_partial_overlap():
    # inter = 5*10 = 50, union = 100 + 100 - 50 = 150 -> 1/3
    iou = _bbox_iou((0, 0, 10, 10), (5, 0, 10, 10))
    assert abs(iou - (1 / 3)) < 1e-6


# -- greedy matching --------------------------------------------------------
def test_match_carries_to_overlapping_new_face_only():
    old = [{"person_id": 7, "confidence": 1.0, "bbox": (0, 0, 10, 10)}]
    new = [(1, 1, 10, 10), (200, 200, 10, 10)]
    matched = _match_human_labels(old, new, threshold=0.5)
    assert set(matched) == {0}
    assert matched[0]["person_id"] == 7


def test_match_assigns_each_old_label_at_most_once():
    # Two new faces both overlap one old label; only the best-IoU new face wins.
    old = [{"person_id": 7, "confidence": 1.0, "bbox": (0, 0, 10, 10)}]
    new = [(0, 0, 10, 10), (3, 0, 10, 10)]
    matched = _match_human_labels(old, new, threshold=0.5)
    assert set(matched) == {0}  # the exact-overlap face, not the weaker one


# -- DB integration ---------------------------------------------------------
def _face(conn, photo_id, person_id, confidence, bbox):
    x, y, w, h = bbox
    conn.execute(
        "INSERT INTO faces (photo_id, person_id, bbox_x, bbox_y, bbox_w, bbox_h, "
        "confidence) VALUES (?,?,?,?,?,?,?)",
        (photo_id, person_id, x, y, w, h, confidence),
    )
    conn.commit()


def _row(bbox, person_id=None, confidence=None):
    x, y, w, h = bbox
    return {"person_id": person_id, "confidence": confidence,
            "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h}


def test_carry_overrides_auto_guess_with_prior_human_label(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        pid = db.upsert_photo(conn, {"path": "/a.jpg", "filename": "a.jpg"})
        person = db.get_or_create_person(conn, "Stefano")
        _face(conn, pid, person, 1.0, (10, 10, 40, 40))  # human label

        # Re-detection produced the same face (auto-guessed someone else) + a new one.
        rows = [_row((11, 11, 40, 40), person_id=999, confidence=0.7),
                _row((300, 300, 40, 40), person_id=None, confidence=None)]
        _carry_human_labels(conn, pid, rows)

        assert rows[0]["person_id"] == person and rows[0]["confidence"] == 1.0
        assert rows[1]["person_id"] is None  # untouched, no overlapping human label
    finally:
        conn.close()


def test_carry_ignores_auto_labels_and_fresh_photos(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        pid = db.upsert_photo(conn, {"path": "/a.jpg", "filename": "a.jpg"})
        person = db.get_or_create_person(conn, "Auto")
        _face(conn, pid, person, 0.7, (10, 10, 40, 40))  # an *auto* guess, not human

        rows = [_row((11, 11, 40, 40), person_id=None, confidence=None)]
        _carry_human_labels(conn, pid, rows)
        assert rows[0]["person_id"] is None  # auto labels are not preserved

        # A photo with no prior faces (fresh index) is a no-op.
        pid2 = db.upsert_photo(conn, {"path": "/b.jpg", "filename": "b.jpg"})
        rows2 = [_row((0, 0, 10, 10), person_id=5, confidence=0.5)]
        _carry_human_labels(conn, pid2, rows2)
        assert rows2[0]["person_id"] == 5  # unchanged
    finally:
        conn.close()
