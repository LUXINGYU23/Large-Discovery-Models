"""``python -m experiment_service --config deployment.json``"""
from __future__ import annotations

import argparse
import logging
import signal
import threading

from .manifest import build_service, load_config
from .server import make_server
from .store import ServiceAlreadyRunning


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiment_service", description="Run the experiment service.")
    parser.add_argument("--config", required=True, help="deployment config JSON")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    try:
        service = build_service(config)
    except ServiceAlreadyRunning as exc:
        parser.exit(1, f"{exc}\n")
    try:
        with make_server(service, config["host"], config["port"], token=config["token"]) as server:
            signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
            service.start_background(config["poll_interval_seconds"])
            print(f"experiment service listening on {server.url}", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    finally:
        service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
