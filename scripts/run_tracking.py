from aerotrack.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["run-tracking", *(__import__("sys").argv[1:])]))
