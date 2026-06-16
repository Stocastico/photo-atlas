"""Video poster-frame ingest — offline via injected stubs, live via ffmpeg.

The metadata parsing (``_parse_probe`` and friends) is pure and tested with
canned ``ffprobe`` JSON. The indexing path is exercised by injecting a stub
poster extractor + probe so the suite never needs ffmpeg; a single live
round-trip is gated on ffmpeg actually being installed.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest
from PIL import Image

from photo_atlas import db, indexer, video
from photo_atlas.config import AtlasConfig


# -- pure ffprobe parsing ---------------------------------------------------
def test_parse_probe_full():
    data = {
        "format": {
            "duration": "12.5",
            "tags": {
                "creation_time": "2019-06-01T12:30:00.000000Z",
                "com.apple.quicktime.location.ISO6709": "+40.7128-074.0060+010.000/",
            },
        },
        "streams": [
            {"codec_type": "audio"},
            {"codec_type": "video", "width": 1920, "height": 1080, "tags": {}},
        ],
    }
    meta = video._parse_probe(data)
    assert meta.taken_at == "2019-06-01T12:30:00"
    assert round(meta.lat, 4) == 40.7128 and round(meta.lon, 4) == -74.0060
    assert meta.width == 1920 and meta.height == 1080
    assert meta.duration == 12.5


def test_parse_probe_stream_creation_time_fallback():
    data = {
        "format": {"tags": {}},
        "streams": [
            {"codec_type": "video", "width": 640, "height": 480,
             "tags": {"creation_time": "2020-01-02 03:04:05"}},
        ],
    }
    meta = video._parse_probe(data)
    assert meta.taken_at == "2020-01-02T03:04:05"
    assert meta.lat is None and meta.lon is None


def test_parse_probe_empty():
    meta = video._parse_probe({})
    assert meta.taken_at is None and meta.width is None and meta.duration is None


def test_parse_creation_time_variants():
    assert video._parse_creation_time("2019-06-01T12:30:00Z") == "2019-06-01T12:30:00"
    assert video._parse_creation_time("2019-06-01T12:30:00.123456Z") == "2019-06-01T12:30:00"
    assert video._parse_creation_time("2019-06-01T12:30:00+02:00") == "2019-06-01T12:30:00"
    assert video._parse_creation_time("not a date") is None


def test_parse_iso6709():
    assert video._parse_iso6709("+40.7128-074.0060+010.000/")[0] == pytest.approx(40.7128)
    assert video._parse_iso6709("+40.7128-074.0060/")[1] == pytest.approx(-74.0060)
    assert video._parse_iso6709("+95.0-074.0/") == (None, None)  # latitude out of range
    assert video._parse_iso6709("garbage") == (None, None)


# -- indexing a video (stubbed poster + probe) ------------------------------
def _stub_extract(path, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), (10, 120, 200)).save(dest, "JPEG")
    return dest


def _make_video_file(tmp_path, name="clip.mp4"):
    path = tmp_path / name
    path.write_bytes(b"\x00\x01\x02not-a-real-video")
    return path


def test_index_video_persists_row_and_derivatives(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    try:
        vid = _make_video_file(tmp_path)
        sha1 = "abc123" + "0" * 34
        meta = video.VideoMeta(
            taken_at="2021-07-04T18:00:00", lat=48.85, lon=2.35, width=1280, height=720
        )
        pid = indexer.index_video(
            conn, config, vid, sha1=sha1,
            extract_poster=_stub_extract, probe_metadata=lambda p: meta,
        )
        conn.commit()
        row = conn.execute("SELECT * FROM photos WHERE id=?", (pid,)).fetchone()
        assert row["is_video"] == 1
        assert row["taken_at"] == "2021-07-04T18:00:00"
        assert row["taken_source"] == "video"
        assert row["lat"] == 48.85 and row["lon"] == 2.35
        assert row["width"] == 1280 and row["height"] == 720
        assert row["path"] == str(vid.resolve())  # the original (playable) file
        # Poster + thumbnail were generated.
        assert indexer.poster_path_for(config, sha1).exists()
        assert row["thumb_path"] and (tmp_path / "lib").exists()
        assert row["face_count"] == 0
    finally:
        conn.close()


def test_index_video_taken_at_falls_back_to_mtime(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    try:
        vid = _make_video_file(tmp_path)
        pid = indexer.index_video(
            conn, config, vid, sha1="d" * 40,
            extract_poster=_stub_extract, probe_metadata=lambda p: video.VideoMeta(),
        )
        row = conn.execute(
            "SELECT taken_at, taken_source FROM photos WHERE id=?", (pid,)
        ).fetchone()
        assert row["taken_source"] == "mtime"
        assert row["taken_at"] is not None
    finally:
        conn.close()


# -- the indexing walk (monkeypatched ffmpeg) -------------------------------
def _seed_tree(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    Image.new("RGB", (32, 32), (200, 50, 50)).save(src / "photo.jpg", "JPEG")
    (src / "movie.mp4").write_bytes(b"\x00video-bytes")
    return src


def test_index_path_indexes_videos_when_ffmpeg_present(tmp_path, monkeypatch):
    monkeypatch.setattr(indexer.video, "ffmpeg_available", lambda: True)
    monkeypatch.setattr(indexer.video, "extract_poster", _stub_extract)
    monkeypatch.setattr(
        indexer.video, "probe_metadata",
        lambda p: video.VideoMeta(taken_at="2022-05-01T09:00:00", width=800, height=600),
    )
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    stats = indexer.index_path(config, _seed_tree(tmp_path), backend_name="none")
    assert stats.videos == 1
    conn = db.connect(config.db_path)
    try:
        videos = conn.execute("SELECT filename FROM photos WHERE is_video=1").fetchall()
        photos = conn.execute("SELECT filename FROM photos WHERE is_video=0").fetchall()
        assert [r["filename"] for r in videos] == ["movie.mp4"]
        assert [r["filename"] for r in photos] == ["photo.jpg"]
    finally:
        conn.close()


def test_index_path_skips_videos_without_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(indexer.video, "ffmpeg_available", lambda: False)
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    stats = indexer.index_path(config, _seed_tree(tmp_path), backend_name="none")
    assert stats.videos == 1
    conn = db.connect(config.db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM photos WHERE is_video=1").fetchone()[0] == 0
    finally:
        conn.close()


# -- media endpoints serve the video's poster derivatives -------------------
def test_api_serves_video_media(tmp_path):
    config = AtlasConfig(home=tmp_path / "lib").ensure_dirs()
    conn = db.connect(config.db_path)
    vid = _make_video_file(tmp_path)
    pid = indexer.index_video(
        conn, config, vid, sha1="e" * 40,
        extract_poster=_stub_extract, probe_metadata=lambda p: video.VideoMeta(),
    )
    conn.commit()
    conn.close()
    from fastapi.testclient import TestClient

    from photo_atlas.api import create_app

    client = TestClient(create_app(config))
    # Grid payload flags the row as a video.
    grid = client.get("/api/photos").json()["photos"]
    assert grid[0]["is_video"] == 1
    # Thumb (default + retina) and preview come from the poster; image is the file.
    assert client.get(f"/api/thumb/{pid}").status_code == 200
    assert client.get(f"/api/thumb/{pid}?size=640").status_code == 200
    assert client.get(f"/api/preview/{pid}").status_code == 200
    assert client.get(f"/api/image/{pid}").status_code == 200


# -- optional live round-trip (only when ffmpeg is installed) ---------------
@pytest.mark.skipif(not video.ffmpeg_available(), reason="ffmpeg/ffprobe not installed")
def test_live_ffmpeg_poster_and_probe(tmp_path):
    clip = tmp_path / "gen.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=duration=1:size=320x240:rate=10", str(clip)],
        check=True, capture_output=True, timeout=60,
    )
    dest = tmp_path / "poster.jpg"
    video.extract_poster(clip, dest)
    assert dest.exists() and dest.stat().st_size > 0
    with Image.open(dest) as img:
        assert img.size == (320, 240)
    meta = video.probe_metadata(clip)
    assert meta.width == 320 and meta.height == 240
    assert meta.duration == pytest.approx(1.0, abs=0.3)


def test_ffmpeg_available_returns_bool():
    assert isinstance(video.ffmpeg_available(), bool)
    # Sanity: the helper agrees with shutil on the configured binaries.
    assert video.ffmpeg_available() == (
        shutil.which(video.FFMPEG) is not None and shutil.which(video.FFPROBE) is not None
    )
