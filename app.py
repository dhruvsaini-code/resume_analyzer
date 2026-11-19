"""Streamlit, Web Server & CLI Launcher for AI Resume Analyzer.

Run options:
1. Web Browser: `python3 app.py` (auto-selects free port starting at 8501)
2. Streamlit:   `streamlit run app.py`
3. CLI Report:  `python3 app.py --cli sample_data/resume.txt sample_data/job_description.txt`
4. Benchmark:   `python3 app.py --benchmark sample_data/job_description.txt`
"""

import argparse
import json
import os
import socket
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_open_port(preferred: int = 8501, attempts: int = 10) -> int:
    for port in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


def _run_web_server(preferred_port: int = 8501, open_browser: bool = True) -> None:
    import http.server
    import socketserver
    import webbrowser

    port = _find_open_port(preferred_port)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=BASE_DIR, **kwargs)

    class ReusableThreadingServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True
        daemon_threads = True

    url = f"http://localhost:{port}/index.html"
    print(f"Starting AI Resume Analyzer at {url}")
    if open_browser:
        webbrowser.open(url)

    with ReusableThreadingServer(("", port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
            sys.exit(0)


def _run_cli(resume_path: str, jd_path: str, as_json: bool = False) -> None:
    from analyzer import evaluate_resume, extract_text, generate_text_report, result_to_dict

    with open(resume_path, "rb") as f:
        resume_text = extract_text(f.read(), resume_path)
    with open(jd_path, "rb") as f:
        jd_text = extract_text(f.read(), jd_path)

    result = evaluate_resume(resume_text, jd_text)
    if as_json:
        print(json.dumps(result_to_dict(result), indent=2))
    else:
        print(generate_text_report(result))


def _run_benchmark(jd_path: str, csv_path: str) -> None:
    from analyzer import evaluate_csv, extract_text

    with open(jd_path, "rb") as f:
        jd_text = extract_text(f.read(), jd_path)

    summary = evaluate_csv(csv_path, jd_text)
    print(f"Benchmarked {summary['total_resumes']} resumes | Overall Avg Score: {summary['average_score']}/100\n")
    print("Category Averages:")
    for label, avg in summary["category_averages"].items():
        print(f"  - {label:28s}: {avg}/100")
    print("\nTop 10 Matching Candidates:")
    for rank, item in enumerate(summary["top_candidates"], 1):
        print(
            f"  {rank:2d}. [{item['label']:25s}] Score: {item['score']:5.1f} ({item['grade']}) "
            f"| Skills: {item['skill_coverage']:5.1f}% | Top Role: {item['top_role']}"
        )


if __name__ == "__main__" and "streamlit" not in sys.modules:
    parser = argparse.ArgumentParser(description="AI Resume Analyzer Launcher")
    parser.add_argument("--port", type=int, default=8501, help="Preferred HTTP port (default: 8501)")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser tab")
    parser.add_argument("--cli", nargs=2, metavar=("RESUME", "JD"), help="Analyze a resume and JD in CLI mode")
    parser.add_argument("--json", action="store_true", help="Output JSON in --cli mode")
    parser.add_argument("--benchmark", metavar="JD", help="Benchmark sample_resumes.csv against a JD file")
    parser.add_argument(
        "--csv",
        default=os.path.join(BASE_DIR, "sample_data", "sample_resumes.csv"),
        help="CSV path for --benchmark",
    )
    args = parser.parse_args()

    if args.cli:
        _run_cli(args.cli[0], args.cli[1], as_json=args.json)
    elif args.benchmark:
        _run_benchmark(args.benchmark, args.csv)
    else:
        _run_web_server(preferred_port=args.port, open_browser=not args.no_browser)
    sys.exit(0)

# If executed via `streamlit run app.py`
try:
    import streamlit as st
    import streamlit.components.v1 as components

    st.set_page_config(
        page_title="AI Resume Analyzer",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    html_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        st.markdown(
            """
            <style>
                .block-container {
                    padding-top: 1rem;
                    padding-bottom: 1rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                    max-width: 100%;
                }
                header {visibility: hidden;}
                footer {visibility: hidden;}
            </style>
            """,
            unsafe_allow_html=True,
        )
        components.html(html_content, height=1400, scrolling=True)
    else:
        st.error("index.html not found. Please ensure index.html exists in the repository.")
except ImportError:
    pass
