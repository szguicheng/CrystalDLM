"""Run/resume only the frozen subset and export its saved receipts on exit."""
from pathlib import Path
import sys

from batch import DEFAULT_RUN_DIR, GlobalStop, BatchRunner


def main():
    runner = BatchRunner(workers=8)
    try:
        print(runner.run(), flush=True)
    except GlobalStop as exc:
        print('Stopped; committed results retained: ' + str(exc), file=sys.stderr, flush=True)
        return 2
    finally:
        if (Path(DEFAULT_RUN_DIR) / 'annotation.sqlite3').exists():
            print('Saved exports:', runner.export(), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
