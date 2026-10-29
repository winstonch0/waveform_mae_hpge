import hashlib
import os
from pathlib import Path

import requests


RECORD = "8257027"

OUT = (
    Path(os.environ["SCRATCH"])
    / "waveform_mae_hpge_data"
    / "raw_full"
)

OUT.mkdir(parents=True, exist_ok=True)

record = requests.get(
    f"https://zenodo.org/api/records/{RECORD}",
    timeout=60,
)

record.raise_for_status()
record = record.json()


def shard_number(name):
    return int(
        name.replace("MJD_Train_", "")
        .replace(".hdf5", "")
    )


files = [
    f
    for f in record["files"]
    if f["key"].startswith("MJD_Train_")
]

files.sort(
    key=lambda f: shard_number(f["key"])
)


for f in files:

    name = f["key"]

    # We already have Train_0
    if name == "MJD_Train_0.hdf5":
        print(f"Using existing {name}")
        continue

    path = OUT / name
    temp = OUT / f"{name}.part"

    expected_checksum = f.get("checksum", "")
    expected_md5 = (
        expected_checksum.split(":", 1)[1]
        if expected_checksum.startswith("md5:")
        else None
    )

    if path.exists():
        print(f"Already exists: {name}")
        continue

    print(f"\nDownloading {name}")

    url = (
        f["links"].get("content")
        or f["links"].get("self")
    )

    if url is None:
        raise RuntimeError(
            f"No download URL found for {name}. "
            f"Available links: {f['links']}"
        )

    with requests.get(
        url,
        stream=True,
        timeout=120,
    ) as r:

        r.raise_for_status()

        downloaded = 0

        with open(temp, "wb") as out:

            for chunk in r.iter_content(
                chunk_size=16 * 1024 * 1024
            ):

                if not chunk:
                    continue

                out.write(chunk)
                downloaded += len(chunk)

                print(
                    f"\r{name}: "
                    f"{downloaded / 1e9:.2f} GB",
                    end="",
                    flush=True,
                )

    print()

    if expected_md5:

        md5 = hashlib.md5()

        with open(temp, "rb") as inp:
            while True:
                block = inp.read(
                    16 * 1024 * 1024
                )

                if not block:
                    break

                md5.update(block)

        actual = md5.hexdigest()

        if actual != expected_md5:
            raise RuntimeError(
                f"Checksum mismatch for {name}: "
                f"{actual} != {expected_md5}"
            )

        print("Checksum OK")

    temp.rename(path)

    print(f"Saved {path}")


print("\nAll training shards downloaded.")
