"""Near-duplicate & burst grouping.

Covers the perceptual hash (dHash) itself, its storage + backfill, the
temporal+perceptual burst grouping (``search.find_burst_groups``), the hard
delete of redundant shots, and the ``/api/duplicates`` + delete endpoints.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from photo_atlas import db, indexer, metadata, search
from photo_atlas.config import AtlasConfig


# -- dHash (perceptual hash) ------------------------------------------------
def _solid(level: int = 10) -> Image.Image:
    return Image.new("L", (9, 8), level).convert("RGB")


def _grid(rows: list[list[int]]) -> Image.Image:
    """Build a 9x8 greyscale image from an explicit luminance grid (8 rows x 9)."""
    img = Image.new("L", (9, 8))
    img.putdata([v for row in rows for v in row])
    return img.convert("RGB")


# A high-contrast alternating pattern: adjacent comparisons give a mix of bits,
# so its dHash has roughly half its bits set (far from a flat image's all-zeros).
_STRIPES = [[0, 255, 0, 255, 0, 255, 0, 255, 0] for _ in range(8)]


def test_dhash_is_16_hex_chars_and_stable():
    h = metadata.dhash(_grid(_STRIPES))
    assert isinstance(h, str)
    assert len(h) == 16  # 64-bit hash, hex
    int(h, 16)  # parses as hex
    assert metadata.dhash(_grid(_STRIPES)) == h  # deterministic


def test_dhash_distance_small_for_similar_large_for_different():
    nudged_rows = [list(r) for r in _STRIPES]
    nudged_rows[0][1] = 0  # flip a single cell → a bit or two changes
    base = metadata.dhash(_grid(_STRIPES))
    nudged = metadata.dhash(_grid(nudged_rows))
    flat = metadata.dhash(_solid())
    assert search.phash_distance(base, base) == 0
    assert 0 < search.phash_distance(base, nudged) < 10
    assert search.phash_distance(base, flat) > 20


def test_phash_distance_handles_missing():
    # A missing hash on either side is "infinitely far" (never groups).
    assert search.phash_distance(None, "0" * 16) == 64
    assert search.phash_distance("0" * 16, None) == 64


# -- burst grouping ---------------------------------------------------------
def _insert(conn, path, taken_at, phash, **cols):
    cols.setdefault("filename", path.rsplit("/", 1)[-1])
    pid = db.upsert_photo(conn, {"path": path, "taken_at": taken_at, **cols})
    db.set_phash(conn, pid, phash)
    return pid


def test_groups_a_near_identical_burst(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        # Five near-identical frames a couple of seconds apart.
        base = 0x0F0F0F0F0F0F0F0F
        ids = []
        for i in range(5):
            ids.append(
                _insert(
                    conn, f"/burst{i}.jpg", f"2021-06-01T10:00:0{i}",
                    f"{base ^ i:016x}", width=4000, height=3000,
                )
            )
        # A lone, unrelated shot an hour later.
        _insert(conn, "/other.jpg", "2021-06-01T11:00:00", "ffffffffffffffff")
        groups = search.find_burst_groups(conn, max_distance=6, max_gap_seconds=10)
        assert len(groups) == 1
        g = groups[0]
        assert g["count"] == 5
        assert set(p["id"] for p in g["photos"]) == set(ids)
        assert g["cover_id"] in ids
    finally:
        conn.close()


def test_time_gap_splits_otherwise_identical_frames(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        # Identical hash, but far apart in time → not a burst.
        _insert(conn, "/a.jpg", "2021-06-01T10:00:00", "0f0f0f0f0f0f0f0f")
        _insert(conn, "/b.jpg", "2021-06-01T10:05:00", "0f0f0f0f0f0f0f0f")
        groups = search.find_burst_groups(conn, max_distance=6, max_gap_seconds=10)
        assert groups == []
    finally:
        conn.close()


def test_visually_different_close_in_time_not_grouped(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        _insert(conn, "/a.jpg", "2021-06-01T10:00:00", "0000000000000000")
        _insert(conn, "/b.jpg", "2021-06-01T10:00:02", "ffffffffffffffff")
        groups = search.find_burst_groups(conn, max_distance=6, max_gap_seconds=10)
        assert groups == []
    finally:
        conn.close()


def test_min_group_drops_pairs(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        _insert(conn, "/a.jpg", "2021-06-01T10:00:00", "0f0f0f0f0f0f0f0f")
        _insert(conn, "/b.jpg", "2021-06-01T10:00:01", "0f0f0f0f0f0f0f0e")
        assert search.find_burst_groups(conn, min_group=3) == []
        assert len(search.find_burst_groups(conn, min_group=2)) == 1
    finally:
        conn.close()


def test_cover_prefers_favorite_then_resolution(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        big = _insert(conn, "/big.jpg", "2021-06-01T10:00:00", "0f0f0f0f0f0f0f0f",
                      width=4000, height=3000)
        small = _insert(conn, "/small.jpg", "2021-06-01T10:00:01", "0f0f0f0f0f0f0f0e",
                        width=1000, height=800)
        # By resolution alone, ``big`` wins.
        g = search.find_burst_groups(conn)[0]
        assert g["cover_id"] == big
        # A favorite overrides resolution.
        db.set_favorite(conn, small, True)
        g = search.find_burst_groups(conn)[0]
        assert g["cover_id"] == small
    finally:
        conn.close()


# -- indexing / backfill / delete -------------------------------------------
def _photos_dir(tmp_path) -> Path:
    d = tmp_path / "photos"
    d.mkdir()
    # Two visually distinct, valid JPEGs.
    _grid(_STRIPES).resize((64, 64)).save(d / "a.jpg")
    Image.new("RGB", (64, 64), (200, 30, 30)).save(d / "b.jpg")
    return d


def test_index_computes_phash(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    indexer.index_path(config, _photos_dir(tmp_path), backend_name="none", geocode=False)
    conn = db.connect(config.db_path)
    try:
        hashes = [r[0] for r in conn.execute("SELECT phash FROM photos")]
        assert len(hashes) == 2
        assert all(h is not None and len(h) == 16 for h in hashes)
    finally:
        conn.close()


def test_backfill_phashes(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    indexer.index_path(config, _photos_dir(tmp_path), backend_name="none", geocode=False)
    conn = db.connect(config.db_path)
    conn.execute("UPDATE photos SET phash=NULL")  # simulate a pre-phash catalog
    conn.commit()
    conn.close()

    assert indexer.backfill_phashes(config) == 2
    conn = db.connect(config.db_path)
    try:
        assert all(r[0] is not None for r in conn.execute("SELECT phash FROM photos"))
        # Idempotent: nothing left to do on a second pass (without --recompute).
        assert indexer.backfill_phashes(config) == 0
        assert indexer.backfill_phashes(config, recompute=True) == 2
    finally:
        conn.close()


def test_delete_photos_removes_rows_and_files(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    photos = _photos_dir(tmp_path)
    indexer.index_path(config, photos, backend_name="none", geocode=False)
    conn = db.connect(config.db_path)
    rows = {r["filename"]: r["id"] for r in conn.execute("SELECT id, filename FROM photos")}
    conn.close()

    target = rows["a.jpg"]
    result = indexer.delete_photos(config, [target])
    assert result["rows"] == 1 and result["files"] == 1
    assert not (photos / "a.jpg").exists()  # source file is gone
    assert (photos / "b.jpg").exists()  # the other is untouched
    conn = db.connect(config.db_path)
    try:
        remaining = [r[0] for r in conn.execute("SELECT id FROM photos")]
        assert target not in remaining and len(remaining) == 1
    finally:
        conn.close()


def test_delete_photos_empty_is_noop(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    result = indexer.delete_photos(config, [])
    assert result["rows"] == 0 and result["files"] == 0


def test_malformed_timestamp_kept_as_singleton(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        # Two visually identical frames, but one carries a non-null yet unparseable
        # taken_at (a garbled stored value). It survives the SQL filter (taken_at
        # IS NOT NULL) but can't be placed on the timeline, so it never groups.
        _insert(conn, "/a.jpg", "not-a-timestamp", "0f0f0f0f0f0f0f0f")
        _insert(conn, "/b.jpg", "also bogus", "0f0f0f0f0f0f0f0f")
        assert search.find_burst_groups(conn) == []
    finally:
        conn.close()


def test_hidden_and_undated_excluded(tmp_path):
    conn = db.connect(tmp_path / "s.db")
    try:
        a = _insert(conn, "/a.jpg", "2021-06-01T10:00:00", "0f0f0f0f0f0f0f0f")
        _insert(conn, "/b.jpg", "2021-06-01T10:00:01", "0f0f0f0f0f0f0f0e")
        _insert(conn, "/undated.jpg", None, "0f0f0f0f0f0f0f0f")
        # Hide one of the burst → the remaining single shot isn't a group.
        db.set_hidden_bulk(conn, [a], True)
        assert search.find_burst_groups(conn) == []
    finally:
        conn.close()


# -- API --------------------------------------------------------------------
def _client(config):
    from fastapi.testclient import TestClient

    from photo_atlas.api import create_app

    return TestClient(create_app(config))


def test_api_duplicates(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    base = 0x0F0F0F0F0F0F0F0F
    for i in range(3):
        _insert(conn, f"/burst{i}.jpg", f"2021-06-01T10:00:0{i}", f"{base ^ i:016x}")
    conn.commit()
    conn.close()

    data = _client(config).get("/api/duplicates").json()
    assert data["count"] == 1
    assert data["redundant"] == 2  # one cover kept, two redundant
    assert data["groups"][0]["count"] == 3


def test_api_delete_photos(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    photos = _photos_dir(tmp_path)
    indexer.index_path(config, photos, backend_name="none", geocode=False)
    client = _client(config)
    conn = db.connect(config.db_path)
    target = conn.execute("SELECT id FROM photos WHERE filename='a.jpg'").fetchone()[0]
    conn.close()

    resp = client.post("/api/photos/delete", json={"ids": [target]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] and body["rows"] == 1 and body["files"] == 1
    assert not (photos / "a.jpg").exists()
    assert client.get(f"/api/photos/{target}").status_code == 404


def test_api_hide_rest_via_bulk(tmp_path):
    # "Hide the rest" reuses the existing bulk action — verify the integration.
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    base = 0x0F0F0F0F0F0F0F0F
    ids = [
        _insert(conn, f"/burst{i}.jpg", f"2021-06-01T10:00:0{i}", f"{base ^ i:016x}")
        for i in range(3)
    ]
    conn.commit()
    conn.close()
    client = _client(config)

    rest = ids[1:]
    hidden = client.post("/api/photos/bulk", json={"ids": rest, "action": "hide"})
    assert hidden.json()["updated"] == 2
    # Hidden shots drop out of the group, leaving the lone cover → no group.
    assert client.get("/api/duplicates").json()["count"] == 0
