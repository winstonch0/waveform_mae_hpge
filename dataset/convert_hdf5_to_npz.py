#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


# Fields expected/used by the Majorana repo.
METADATA_FIELDS = [
    "tp0",
    "energy_label",
    "detector",
    "id",
    "psd_label_dcr",
    "psd_label_high_avse",
    "psd_label_low_avse",
    "psd_label_lq",
    "run_number",
]


def inspect_hdf5(f):
    """Print the HDF5 schema."""
    print("\nHDF5 contents:")
    for key in f.keys():
        obj = f[key]
        if isinstance(obj, h5py.Dataset):
            print(f"  {key:25s} shape={obj.shape} dtype={obj.dtype}")
        else:
            print(f"  {key:25s} <group>")


def scalar_or_array(value):
    """
    Convert HDF5 values into a form np.savez can store cleanly.
    Scalars remain scalars; arrays remain arrays.
    """
    arr = np.asarray(value)

    if arr.ndim == 0:
        return arr.item()

    return arr


def convert(
    input_path,
    output_dir,
    max_events=None,
    start_index=0,
    overwrite=False,
):
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as f:

        inspect_hdf5(f)

        # The public Majorana HDF5 uses raw_waveform.
        if "raw_waveform" not in f:
            raise KeyError(
                "Expected HDF5 dataset 'raw_waveform', "
                f"but found: {list(f.keys())}"
            )

        n_total = len(f["raw_waveform"])

        if start_index < 0 or start_index >= n_total:
            raise ValueError(
                f"start_index={start_index} is outside dataset range "
                f"[0, {n_total - 1}]"
            )

        if max_events is None:
            stop_index = n_total
        else:
            stop_index = min(start_index + max_events, n_total)

        print("\nConversion:")
        print(f"  Source:       {input_path}")
        print(f"  Output:       {output_dir}")
        print(f"  Total events: {n_total}")
        print(f"  Start index:  {start_index}")
        print(f"  Stop index:   {stop_index}")
        print(f"  Converting:   {stop_index - start_index}")

        # Check metadata dimensions.
        available_metadata = []

        for key in METADATA_FIELDS:
            if key not in f:
                print(f"WARNING: field '{key}' not present")
                continue

            if len(f[key]) != n_total:
                raise ValueError(
                    f"Field '{key}' has {len(f[key])} entries, "
                    f"but raw_waveform has {n_total}"
                )

            available_metadata.append(key)

        converted = 0
        skipped = 0

        for i in range(start_index, stop_index):

            # Deterministic filename.
            output_file = output_dir / f"event_{i:08d}.npz"

            if output_file.exists() and not overwrite:
                skipped += 1
                continue

            waveform = np.asarray(f["raw_waveform"][i])

            # Basic integrity checks.
            if waveform.ndim != 1:
                raise ValueError(
                    f"Event {i}: waveform is not 1D. "
                    f"Shape = {waveform.shape}"
                )

            if not np.all(np.isfinite(waveform)):
                raise ValueError(
                    f"Event {i}: waveform contains NaN or Inf"
                )

            event = {
                # Repo expects this key to be called "waveform".
                "waveform": waveform,
            }

            for key in available_metadata:
                event[key] = scalar_or_array(f[key][i])

            np.savez(output_file, **event)

            converted += 1

            if converted % 100 == 0:
                print(f"  Converted {converted} events...")

    # Write a simple manifest describing this conversion.
    manifest = {
        "source_file": str(input_path.resolve()),
        "output_directory": str(output_dir.resolve()),
        "start_index": start_index,
        "stop_index": stop_index,
        "requested_max_events": max_events,
        "converted_events": converted,
        "skipped_existing_events": skipped,
        "waveform_source_field": "raw_waveform",
        "waveform_output_field": "waveform",
        "metadata_fields": available_metadata,
    }

    manifest_path = output_dir / "conversion_manifest.json"

    with open(manifest_path, "w") as fp:
        json.dump(manifest, fp, indent=2)

    print("\nDone.")
    print(f"  Converted: {converted}")
    print(f"  Skipped:   {skipped}")
    print(f"  Manifest:  {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Majorana HDF5 events into one NPZ file per event."
    )

    parser.add_argument(
        "input",
        help="Input Majorana HDF5 file",
    )

    parser.add_argument(
        "output",
        help="Directory for converted NPZ files",
    )

    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum number of events to convert",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Event index to begin conversion from",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite NPZ files that already exist",
    )

    args = parser.parse_args()

    convert(
        input_path=args.input,
        output_dir=args.output,
        max_events=args.max_events,
        start_index=args.start_index,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
