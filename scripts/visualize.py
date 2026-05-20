from aerotrack.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["visualize", *(__import__("sys").argv[1:])]))
