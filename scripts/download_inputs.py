#!/usr/bin/env python3
"""Download the raw inputs of an input build into work/raw/, at the paths the builds read by default.

    python scripts/download_inputs.py grid [--email ADDRESS --institution NAME] [--name NAME] [--position TEXT]
    python scripts/download_inputs.py grid --check

Run it from the package root. Standard library only. Every file is checked against the SHA-256
of the file the paper was built from; a file already present with that hash is not downloaded
again. A download with any other hash stops the script with an error: the source has changed
and the build would not reproduce the paper.

Target `grid`: the inputs of `python -m src.reach.grid`, which writes the common
5 arcmin grid under work/grid/ (country raster, pixel area, ruminant heads, SPAM production)
read by src/reach/access.py and src/figures/fig2_reach_inputs.py.

  FAO GLW4 2020, Gridded Livestock of the World, dasymetric densities of cattle, buffaloes,
    sheep and goats (heads per km2, 5 arcmin). FAO (2024), CC BY 4.0, FAO Data Catalogue
    https://data.apps.fao.org/catalog/dataset/9d1e149b-d63f-4213-978b-317a8eb42d02
  SPAM 2020 V1r0, yield and harvested area, all technologies (one CSV each). IFPRI (2024),
    Global Spatially-Disaggregated Crop Production Statistics Data for 2020 Version 1.0.0,
    https://doi.org/10.7910/DVN/SWPENT, Harvard Dataverse, dataset version 3.0. CC BY 4.0
    (IFPRI terms of use). Later dataset versions hold SPAM 2020 V2r0 and V2r2, which differ.
    The two zip archives that hold the CSVs are downloaded, the CSV is extracted and the
    archive deleted.
  Natural Earth 1:10m Admin 0 Countries, version 5.1.1 (shapefile), public domain,
    https://www.naturalearthdata.com, from the tagged release of its source repository.

The SPAM dataset has a guestbook: Harvard Dataverse asks every downloader for an email address
and an institution (name and position are optional). Give them with --email and --institution,
or type them when asked; the script sends them to Harvard Dataverse only, with the download
request. Alternatively, download spam2020V1r0_global_yield.csv.zip and
spam2020V1r0_global_harvested_area.csv.zip in a browser from version 3.0 of the dataset
(https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SWPENT&version=3.0),
put them in work/raw/SPAM/ and rerun this script: it checks and extracts them.

Download size for `grid`: about 177 MB (GLW4 43 MB, SPAM archives 124 MB, Natural Earth 10 MB);
about 380 MB on disk afterwards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "work" / "raw"
USER_AGENT = "residue-ruminants-download-inputs/1.0 (Python urllib)"
CHUNK = 1 << 20

GLW_URL = "https://storage.googleapis.com/fao-gismgr-glw4-2020-data/DATA/GLW4-2020/MAPSET/D-DA/GLW4-2020.D-DA.{}.tif"
NE_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.1/10m_cultural/ne_10m_admin_0_countries.{}"
DATAVERSE_FILE = "https://dataverse.harvard.edu/api/access/datafile/{}"
SPAM_PAGE = "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SWPENT&version=3.0"

# Files downloaded as they are: path under work/raw, URL, size in bytes, SHA-256.
DIRECT = {
    "grid": [
        ("GLW4-2020.D-DA.CTL.tif", GLW_URL.format("CTL"), 12667017,
         "2bcbf3b57dec7d6f45f2dcfb9cf61d04f22fae1d636934fb8c41a6a7742be3b5"),
        ("GLW4-2020.D-DA.BFL.tif", GLW_URL.format("BFL"), 6263918,
         "e2c7377b33d14cb7ffdfe71906b2051406de1b8caf4ea9c2dffb1ddac1385ddc"),
        ("GLW4-2020.D-DA.SHP.tif", GLW_URL.format("SHP"), 12175665,
         "42a6e641b7cc6f1beabd535f4afce1aca896f4e6a398d688072ccfe36174a47f"),
        ("GLW4-2020.D-DA.GTS.tif", GLW_URL.format("GTS"), 12071177,
         "f5c0372e0bf9631c284615e256e6dc81b7c2b8ebee6808cf1427c7edc8a9b169"),
        ("natural-earth/ne_10m_admin_0_countries.shp", NE_URL.format("shp"), 8806224,
         "7ce119ef6342e43cff7c0c3004e0911ab7ec1988a14734372031d2012180e7bc"),
        ("natural-earth/ne_10m_admin_0_countries.shx", NE_URL.format("shx"), 2164,
         "ca19ec112d054c77bc8f7ac00e3b110d5dff32cc9bcf4cd1b8b66bdd0f611d32"),
        ("natural-earth/ne_10m_admin_0_countries.dbf", NE_URL.format("dbf"), 878482,
         "c5dbd3dd5fd7e2ef49051fc88562c03819e8ea63a382642df6eadd1243bf4b49"),
        ("natural-earth/ne_10m_admin_0_countries.prj", NE_URL.format("prj"), 145,
         "a02a27b1d1982c8516d83398e85a3c8b1aef1713c13ef4d84d7bde17430c07c4"),
        ("natural-earth/ne_10m_admin_0_countries.cpg", NE_URL.format("cpg"), 5,
         "3ad3031f5503a4404af825262ee8232cc04d4ea6683d42c5dd0a2f2a27ac9824"),
    ],
}

# Files extracted from a Harvard Dataverse zip archive: path under work/raw, Dataverse file id,
# archive name, archive size in bytes, archive MD5 (as published by Dataverse), CSV name in the
# archive, CSV size in bytes, CSV SHA-256.
ARCHIVED = {
    "grid": [
        ("SPAM/spam2020V1r0_global_Y_TA.csv", 10120789, "spam2020V1r0_global_yield.csv.zip", 75610130,
         "ee0b10b9d561b09f7100463125bf4c4b", "spam2020V1r0_global_Y_TA.csv", 176623695,
         "7d17e7330df4a23b4ad4310ad264965037158da57771eabfd7494f3cc7a0a719"),
        ("SPAM/spam2020V1r0_global_harvested_area/spam2020V1r0_global_H_TA.csv", 10120788,
         "spam2020V1r0_global_harvested_area.csv.zip", 48610674,
         "6a00ec59fa72992f97a6ec01f9a1cd20", "spam2020V1r0_global_H_TA.csv", 153579922,
         "aa37f9a2ff9ae3cae585e2425d61aa2c97e3d87b351745c4a0b935ba973bc71f"),
    ],
}

# Environment variables with which the builds read a raw input from elsewhere than work/raw.
OVERRIDES = {"grid": ("RR_SPAM", "RR_NATURAL_EARTH")}


class Mismatch(Exception):
    """A file whose size or hash differs from the one the paper was built from."""


def mb(n):
    if 0 < n < 1e3:
        return f"{n} bytes"
    return f"{n / 1e6:.1f} MB" if n >= 1e5 or n == 0 else f"{n / 1e3:.1f} kB"


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def status(path, size, sha256):
    """'missing', 'ok', or 'differs' for the file at path."""
    if not path.is_file():
        return "missing"
    if path.stat().st_size != size or digest(path) != sha256:
        return "differs"
    return "ok"


def request(url, data=None):
    headers = {"User-Agent": USER_AGENT}
    if data is not None:
        headers["Content-Type"] = "application/json"
    return urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")


def fetch(url, part, size, attempts=3):
    """Stream url into part (size bytes expected); returns its SHA-256 and MD5."""
    for attempt in range(1, attempts + 1):
        try:
            sha, md5, done, shown = hashlib.sha256(), hashlib.md5(), 0, time.monotonic()
            with urllib.request.urlopen(request(url), timeout=120) as response, open(part, "wb") as out:
                for block in iter(lambda: response.read(CHUNK), b""):
                    out.write(block)
                    sha.update(block)
                    md5.update(block)
                    done += len(block)
                    if time.monotonic() - shown > 5:
                        print(f"      {mb(done)} of {mb(size)}", flush=True)
                        shown = time.monotonic()
            return sha.hexdigest(), md5.hexdigest()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            part.unlink(missing_ok=True)
            if isinstance(error, urllib.error.HTTPError) and error.code < 500:
                raise
            if attempt == attempts:
                raise
            print(f"      {error}; retrying ({attempt} of {attempts - 1})", flush=True)
            time.sleep(5 * attempt)


def get_direct(rel, url, size, sha256):
    target = RAW / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    print(f"  {rel}: downloading {mb(size)} from {url}", flush=True)
    got, _ = fetch(url, part, size)
    if got != sha256:
        part.unlink(missing_ok=True)
        raise Mismatch(f"{rel}: the download from {url} has SHA-256 {got}, the paper's file {sha256}")
    part.replace(target)
    return size


class Guestbook:
    """The guestbook answers that Harvard Dataverse requires before a SPAM download."""

    def __init__(self, args):
        self.answers = {k: v for k, v in (("email", args.email), ("institution", args.institution),
                                          ("name", args.name), ("position", args.position)) if v}

    def response(self):
        if not ("email" in self.answers and "institution" in self.answers):
            if not sys.stdin.isatty():
                raise SystemExit(
                    "The SPAM 2020 files come from Harvard Dataverse, whose guestbook for this dataset requires an\n"
                    "email address and an institution. Rerun with --email and --institution, or download the two\n"
                    f"archives in a browser from {SPAM_PAGE}\n"
                    "into work/raw/SPAM/ and rerun.")
            print("  Harvard Dataverse requires an email address and an institution (its guestbook for the SPAM\n"
                  "  dataset) before the download. They are sent to Harvard Dataverse only.")
            for key in ("email", "institution"):
                while not self.answers.get(key):
                    self.answers[key] = input(f"    {key}: ").strip()
        return {"guestbookResponse": self.answers}


def signed_url(file_id, guestbook):
    """Post the guestbook response for one Dataverse file; Dataverse answers with a download link."""
    body = json.dumps(guestbook.response()).encode()
    try:
        with urllib.request.urlopen(request(DATAVERSE_FILE.format(file_id), body), timeout=60) as response:
            reply = json.load(response)
    except urllib.error.HTTPError as error:
        try:
            message = json.load(error).get("message", "")
        except ValueError:
            message = ""
        raise SystemExit(f"Harvard Dataverse refused the download of file {file_id}: HTTP {error.code} {message}")
    url = (reply.get("data") or {}).get("signedUrl")
    if not url:
        raise SystemExit(f"Harvard Dataverse gave no download link for file {file_id}: {reply}")
    return url


def get_archived(rel, file_id, archive, archive_size, archive_md5, member, size, sha256, guestbook):
    target = RAW / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    local = RAW / "SPAM" / archive
    downloaded = 0
    if local.is_file():
        print(f"  {rel}: extracting from {local.relative_to(ROOT)}", flush=True)
        zip_path = local
        if digest(local, "md5") != archive_md5:
            raise Mismatch(f"{local.relative_to(ROOT)}: MD5 {digest(local, 'md5')} differs from the archive in "
                           f"dataset version 3.0 ({archive_md5}); download it from {SPAM_PAGE}")
    else:
        zip_path = local.with_name(archive + ".part")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {rel}: downloading {archive} ({mb(archive_size)}, Harvard Dataverse file {file_id})", flush=True)
        _, md5 = fetch(signed_url(file_id, guestbook), zip_path, archive_size)
        if md5 != archive_md5:
            zip_path.unlink(missing_ok=True)
            raise Mismatch(f"{archive}: the download has MD5 {md5}, Harvard Dataverse publishes {archive_md5}")
        downloaded = archive_size
    part = target.with_name(target.name + ".part")
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = [n for n in z.namelist() if n.rsplit("/", 1)[-1] == member and not n.startswith("__MACOSX")]
            if len(names) != 1:
                raise Mismatch(f"{archive}: {len(names)} entries named {member} (expected 1)")
            h = hashlib.sha256()
            with z.open(names[0]) as src, open(part, "wb") as out:
                for block in iter(lambda: src.read(CHUNK), b""):
                    out.write(block)
                    h.update(block)
        if h.hexdigest() != sha256:
            part.unlink(missing_ok=True)
            raise Mismatch(f"{rel}: {member} in {archive} has SHA-256 {h.hexdigest()}, the paper's file {sha256}")
        part.replace(target)
    finally:
        if zip_path != local:
            zip_path.unlink(missing_ok=True)
    return downloaded


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", choices=sorted(DIRECT), help="the build whose raw inputs to get")
    parser.add_argument("--check", action="store_true", help="only report which files are present and correct")
    parser.add_argument("--email", help="SPAM guestbook: email address (required by Harvard Dataverse)")
    parser.add_argument("--institution", help="SPAM guestbook: institution (required by Harvard Dataverse)")
    parser.add_argument("--name", help="SPAM guestbook: name (optional)")
    parser.add_argument("--position", help="SPAM guestbook: position (optional)")
    args = parser.parse_args()

    direct, archived = DIRECT[args.target], ARCHIVED.get(args.target, [])
    print(f"{args.target}: {len(direct) + len(archived)} files under {RAW.relative_to(ROOT)}/")
    for name in OVERRIDES.get(args.target, ()):
        if os.environ.get(name):
            print(f"  note: {name} is set, so the build reads that input from {os.environ[name]}, not from work/raw")

    needed, problems = [], []
    for item in direct + archived:
        rel, size, sha256 = item[0], item[-2], item[-1]
        state = status(RAW / rel, size, sha256)
        if state == "ok":
            print(f"  {rel}: present, SHA-256 matches")
        elif state == "differs":
            problems.append(f"{rel}: present but its size or SHA-256 differs from the paper's file; "
                            "move it away to download the right one")
        else:
            needed.append(item)
    for p in problems:
        print(f"  {p}")
    if args.check:
        for item in needed:
            print(f"  {item[0]}: missing")
        sys.exit(1 if needed or problems else 0)

    guestbook, fetched, failures = Guestbook(args), 0, []
    for item in needed:
        try:
            if item in direct:
                fetched += get_direct(*item)
            else:
                fetched += get_archived(*item, guestbook)
            print(f"  {item[0]}: SHA-256 matches")
        except Mismatch as error:
            failures.append(str(error))
            print(f"  ERROR {error}", flush=True)
        except urllib.error.URLError as error:
            failures.append(f"{item[0]}: download failed: {error}")
            print(f"  ERROR {item[0]}: download failed: {error}", flush=True)
    if problems or failures:
        print(f"\n{len(problems) + len(failures)} file(s) missing or not as the paper's:")
        for p in problems + failures:
            print(f"  {p}")
        sys.exit(1)
    print(f"Done: {len(needed)} file(s) obtained ({mb(fetched)} downloaded), "
          f"{len(direct) + len(archived) - len(needed)} already present.")
    if args.target == "grid":
        print("Next: python -m src.reach.grid")


if __name__ == "__main__":
    main()
