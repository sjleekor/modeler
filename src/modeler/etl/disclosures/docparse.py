"""DART 공시 원문 zip 읽기와 표(행) 추출.

a·ga 원문은 확장자만 xml이고 실제로는 HTML 표다. 머리에 ``charset=euc-kr``를
선언하지만 실제 바이트는 utf-8과 cp949가 섞여 있다. 선언을 믿지 않고 utf-8을 먼저
시도해 실패하면 cp949로 읽는다.
"""

from __future__ import annotations

import hashlib
import html
import io
import re
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser


@dataclass(frozen=True)
class RawDoc:
    """zip 하나에서 읽은 본문."""

    text: str
    member: str
    encoding: str  # utf-8 · cp949 · replace
    declared_encoding: str | None
    encoding_mismatch: bool
    n_members: int


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_bytes(b: bytes) -> tuple[str, str]:
    """utf-8을 먼저, 안 되면 cp949. 둘 다 안 되면 utf-8 replace(``replace``로 표시)."""
    if b[:3] == b"\xef\xbb\xbf":
        b = b[3:]
    for enc in ("utf-8", "cp949"):
        try:
            return b.decode(enc), enc
        except UnicodeDecodeError:
            pass
    return b.decode("utf-8", "replace"), "replace"


_DECL = re.compile(r"charset\s*=\s*[\"']?\s*([A-Za-z0-9_\-]+)", re.I)
_ENC_ALIAS = {"euc-kr": "cp949", "euckr": "cp949", "ks_c_5601-1987": "cp949", "utf8": "utf-8"}


def _canon(enc: str) -> str:
    e = enc.lower()
    return _ENC_ALIAS.get(e, e)


def read_zip_bytes(data: bytes, rcept_no: str | None = None) -> RawDoc:
    """zip 바이트에서 본문 한 개를 골라 읽는다. 본문은 ``{rcept_no}.xml``을 우선한다."""
    z = zipfile.ZipFile(io.BytesIO(data))
    names = [n for n in z.namelist() if not n.endswith("/")]
    if not names:
        raise ValueError("empty zip")
    member = names[0]
    if rcept_no:
        for n in names:
            if n.rsplit("/", 1)[-1] == f"{rcept_no}.xml":
                member = n
                break
    text, enc = decode_bytes(z.read(member))
    m = _DECL.search(text[:2000])
    declared = _canon(m.group(1)) if m else None
    mismatch = bool(declared and enc in ("utf-8", "cp949") and _canon(enc) != declared)
    return RawDoc(text, member, enc, declared, mismatch, len(names))


def norm(s: str) -> str:
    """공백 정리. ``&cr;``는 줄바꿈 표기라 공백으로 바꾼다."""
    s = html.unescape(s).replace("\xa0", " ").replace("&cr;", " ")
    return re.sub(r"\s+", " ", s).strip()


def key(s: str) -> str:
    """라벨 비교용: 모든 공백 제거."""
    return re.sub(r"\s+", "", s)


class _RowParser(HTMLParser):
    """tr 단위로 셀 텍스트 목록을 모은다. 중첩 표의 안쪽 셀은 바깥 셀 텍스트에 합쳐진다."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.cur: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.cur = []
        elif tag in ("td", "th"):
            self.cell = []
        elif tag == "br" and self.cell is not None:
            self.cell.append("\n")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None and self.cur is not None:
            self.cur.append(norm("".join(self.cell)))
            self.cell = None
        elif tag == "tr" and self.cur is not None:
            if self.cur:
                self.rows.append(self.cur)
            self.cur = None

    def handle_data(self, d):
        if self.cell is not None:
            self.cell.append(d)


def rows_of(text: str) -> list[list[str]]:
    i = text.lower().find("<body")
    p = _RowParser()
    p.feed(text[i:] if i >= 0 else text)
    p.close()
    return p.rows
