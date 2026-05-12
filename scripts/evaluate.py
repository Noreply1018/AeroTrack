from aerotrack.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["run-experiment", *(__import__("sys").argv[1:]), "--preflight-only"]))
