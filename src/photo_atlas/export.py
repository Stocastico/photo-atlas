"""Export named-person labels as portable XMP sidecars.

Person names live only in the SQLite catalog, so the (often considerable) work
of naming faces is neither backed up with the photos nor portable to other
tools. :func:`write_xmp_sidecars` writes a ``<photo>.xmp`` next to each photo
that has named people, recording the names as Dublin Core keywords
(``dc:subject``) plus a ``People|<name>`` hierarchical keyword. Originals are
never modified, and apps like digiKam, Lightroom and Adobe Bridge read these
sidecars, so the labels survive a catalog loss and travel with the library.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from xml.sax.saxutils import escape

from . import db
from .config import AtlasConfig

_XMP_TEMPLATE = """<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:lr="http://ns.adobe.com/lightroom/1.0/">
   <dc:subject>
    <rdf:Bag>
{subjects}
    </rdf:Bag>
   </dc:subject>
   <lr:hierarchicalSubject>
    <rdf:Bag>
{hierarchical}
    </rdf:Bag>
   </lr:hierarchicalSubject>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
"""


def _xmp_document(names: list[str]) -> str:
    """Render a minimal, valid XMP packet tagging the given people."""

    subjects = "\n".join(f"     <rdf:li>{escape(n)}</rdf:li>" for n in names)
    hierarchical = "\n".join(
        f"     <rdf:li>People|{escape(n)}</rdf:li>" for n in names
    )
    return _XMP_TEMPLATE.format(subjects=subjects, hierarchical=hierarchical)


def _photos_with_people(config: AtlasConfig) -> "OrderedDict[str, list[str]]":
    """Map each photo path to the sorted, de-duplicated names it contains."""

    conn = db.connect(config.db_path)
    try:
        rows = conn.execute(
            "SELECT p.path AS path, pr.name AS name "
            "FROM photos p "
            "JOIN faces f ON f.photo_id = p.id "
            "JOIN persons pr ON pr.id = f.person_id "
            "ORDER BY p.id"
        ).fetchall()
    finally:
        conn.close()

    grouped: "OrderedDict[str, set[str]]" = OrderedDict()
    for row in rows:
        grouped.setdefault(row["path"], set()).add(row["name"])
    return OrderedDict((path, sorted(names)) for path, names in grouped.items())


def write_xmp_sidecars(config: AtlasConfig, dest: Path | None = None) -> dict[str, int]:
    """Write an XMP sidecar per photo that has named people.

    By default the sidecar is written next to the original as
    ``<photo>.xmp`` (the exiftool/digiKam convention). Pass ``dest`` to collect
    them in one directory instead, leaving the photo tree untouched.
    """

    grouped = _photos_with_people(config)
    written = 0
    out_dir = Path(dest) if dest is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    for photo_path, names in grouped.items():
        doc = _xmp_document(names)
        if out_dir is not None:
            sidecar = out_dir / (Path(photo_path).name + ".xmp")
        else:
            sidecar = Path(photo_path + ".xmp")
        try:
            sidecar.write_text(doc, encoding="utf-8")
            written += 1
        except OSError:
            # E.g. the original tree is read-only or has been moved away.
            continue
    return {"photos": len(grouped), "written": written}
