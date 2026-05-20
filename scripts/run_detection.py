from aerotrack.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["run-detection", *(__import__("sys").argv[1:])]))
