"""Start the dashboard backend server."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Start dashboard backend")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--db", default="smallest_llm.db")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    # Set DB path before importing the app
    import src.server.app as server_app
    server_app.DB_PATH = args.db

    import uvicorn
    uvicorn.run(
        "src.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
