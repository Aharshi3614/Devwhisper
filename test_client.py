"""
test_client.py — Standalone CLI test client for DevWhisper.

This module provides a command-line interface for testing the DevWhisper
pipeline end-to-end without requiring a live Vapi voice connection. It sends
synthetic tool-call payloads to the /webhook endpoint and displays responses.

Usage:
    python test_client.py                    # Interactive mode
    python test_client.py -q "your query"    # One-shot query
    python test_client.py -q "your query" -u http://remote:8000/webhook

Features:
    - Interactive REPL-style querying
    - One-shot query via --query / -q flag
    - Custom webhook URL via --url / -u flag
    - Graceful handling of KeyboardInterrupt and network errors
"""

import argparse
import requests
import json
import uuid
import sys


def send_query(query: str, webhook_url: str) -> None:
    """
    Send a single query to the DevWhisper webhook endpoint.

    Constructs a synthetic Vapi "tool-calls" payload with a single
    query_codebase function call, POSTs it to the webhook, and prints
    the response.

    Args:
        query: The natural language or code query string.
        webhook_url: Full URL of the DevWhisper /webhook endpoint.

    Side effects:
        Prints the response to stdout or an error message to stderr.
    """
    payload = {
        "message": {
            "type": "tool-calls",
            "toolCalls": [
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": "query_codebase",
                        "arguments": json.dumps({"query": query})
                    }
                }
            ]
        }
    }

    print("\nSending query to DevWhisper...")
    response = None
    try:
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()

        # DevWhisper's webhook returns JSON with a 'results' list
        data = response.json()
        results = data.get("results", [])

        if not results:
            print("No results returned from DevWhisper.")
            return

        for res in results:
            print(f"\nResponse:\n{res.get('result')}\n")
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with DevWhisper: {e}")
        if response is not None and hasattr(response, "text"):
            print(f"Server replied: {response.text}")


def main() -> None:
    """
    CLI entry point for the DevWhisper test client.

    Parses command-line arguments and either runs in interactive mode
    or sends a single query.
    """
    parser = argparse.ArgumentParser(
        description="Standalone Test Client for DevWhisper"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Text query to send to DevWhisper",
        required=False,
    )
    parser.add_argument(
        "--url", "-u",
        type=str,
        default="http://localhost:8000/webhook",
        help="DevWhisper webhook URL (default: http://localhost:8000/webhook)",
    )

    args = parser.parse_args()

    # One-shot mode
    if args.query:
        send_query(args.query, args.url)
        return

    # Interactive mode
    print("=== DevWhisper Standalone Test Client ===")
    print("Type your query below. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            query = input("\nEnter query: ").strip()
            if not query:
                continue
            if query.lower() in ['exit', 'quit']:
                break
            send_query(query, args.url)
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
        except EOFError:
            break


if __name__ == "__main__":
    main()
    
