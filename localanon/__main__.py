from .cli import main

if __name__ == "__main__":     # python -m localanon; importing the module (spawn, tools) does not run main
    raise SystemExit(main())
