from pathlib import Path
import csv
import argparse

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--output", default="training_metrics")
    args = parser.parse_args()

    logdir = Path(args.logdir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    event_files = list(logdir.rglob("events.out.tfevents.*"))

    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under {logdir}")

    print(f"Found {len(event_files)} event file(s).")

    for event_file in event_files:
        print(f"\nReading: {event_file}")

        ea = EventAccumulator(str(event_file))
        ea.Reload()

        scalar_tags = ea.Tags().get("scalars", [])

        print("Scalar tags:")
        for tag in scalar_tags:
            print("  ", tag)

        for tag in scalar_tags:
            events = ea.Scalars(tag)

            safe_name = tag.replace("/", "_")
            csv_path = output / f"{safe_name}.csv"

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "wall_time", "value"])

                for event in events:
                    writer.writerow([
                        event.step,
                        event.wall_time,
                        event.value,
                    ])

            print(f"Saved {tag} -> {csv_path}")


if __name__ == "__main__":
    main()
