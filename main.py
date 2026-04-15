import os
import csv
import smtplib
import requests
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
import schedule

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
_gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GMAIL_ADDRESS = os.environ["GMAIL_ADDRESS"]
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
RECIPIENT_EMAIL = os.environ["RECIPIENT_EMAIL"]

TOPICS = {
    "AI for Education": [
        "LLM tutoring students", "AI adaptive learning",
        "large language model education", "AI writing feedback students",
        "generative AI student learning outcomes"
    ],
    "AI for Health": [
        "large language model clinical", "AI medical diagnosis",
        "LLM radiology pathology", "AI mental health intervention",
        "foundation model healthcare"
    ],
    "Model Behavior": [
        "LLM alignment", "AI misalignment", "agentic AI safety",
        "large language model evaluation", "LLM robustness",
        "AI deception", "AI goal misgeneralization", "LLM safety behavior",
        "AI agent misalignment", "language model alignment"
    ],
    "AI Ethics": [
        "AI ethics empirical", "algorithmic bias language model",
        "AI fairness machine learning", "LLM harmful outputs",
        "AI discrimination societal impact"
    ],
    "AI Research Advances": [
        "large language model novel architecture", "LLM reasoning new method",
        "multimodal AI new capability", "reinforcement learning from human feedback",
        "frontier AI model capabilities"
    ],
    "Major AI Players": [
        "OpenAI GPT research", "Google DeepMind research",
        "Anthropic Claude research", "Meta LLaMA research",
        "Microsoft AI research paper"
    ],
    "Human-AI Interaction & Psychology": [
        "chatbot user psychology", "AI companion user loneliness",
        "human trust in AI systems", "anthropomorphism chatbot",
        "LLM user behavior study", "AI chatbot mental health user",
        "human perception AI agent"
    ],
}

TOPIC_DESCRIPTIONS = {
    "AI for Education": "Research on using AI, LLMs, or machine learning to support student learning, tutoring, or educational outcomes.",
    "AI for Health": "Research applying AI or large language models to medical diagnosis, clinical decision-making, or healthcare.",
    "Model Behavior": "Research on how AI and LLM models behave, including alignment, misalignment, safety, deception, and robustness.",
    "AI Ethics": "Research on ethical issues, bias, fairness, or societal harms caused by AI and machine learning systems.",
    "AI Research Advances": "New research advancing the capabilities of AI, LLMs, or machine learning — novel architectures, training methods, or benchmarks. Must be about artificial intelligence or machine learning, not other fields.",
    "Major AI Players": "Research papers published by or about major AI organizations such as OpenAI, Google DeepMind, Anthropic, Meta AI, or Microsoft Research.",
    "Human-AI Interaction & Psychology": "Research on how people psychologically perceive, trust, relate to, or are affected by AI systems and chatbots.",
}

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def fetch_papers(query: str, days_back: int = 10, limit: int = 10) -> list[dict]:
    since = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    params = {
        "query": query,
        "fields": "title,authors,year,publicationDate,publicationTypes,venue,externalIds,abstract,url,citationCount,influentialCitationCount",
        "publicationDateOrYear": f"{since}:",
        "limit": limit,
    }
    try:
        response = requests.get(SEMANTIC_SCHOLAR_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])
    except Exception:
        return []


def is_valid_paper(paper: dict) -> bool:
    return bool(
        paper.get("title")
        and paper.get("abstract")
        and paper.get("publicationTypes")
        and paper.get("publicationDate")
    )


def deduplicate(papers: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for p in papers:
        pid = p.get("paperId") or p.get("title")
        if pid and pid not in seen:
            seen.add(pid)
            unique.append(p)
    return unique


def get_paper_url(paper: dict) -> str:
    ext = paper.get("externalIds", {})
    if ext.get("DOI"):
        return f"https://doi.org/{ext['DOI']}"
    if ext.get("ArXiv"):
        return f"https://arxiv.org/abs/{ext['ArXiv']}"
    return paper.get("url", "")


def _gemini_generate(prompt: str, retries: int = 5) -> str:
    delay = 10
    for attempt in range(retries):
        try:
            response = _gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite", contents=prompt
            )
            return response.text.strip()
        except genai_errors.ServerError as e:
            if e.status_code == 503 and attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                raise


def is_relevant(paper: dict, topic: str) -> bool:
    description = TOPIC_DESCRIPTIONS.get(topic, topic)
    prompt = (
        f"Topic: {description}\n\n"
        f"Title: {paper['title']}\n\n"
        f"Abstract: {paper['abstract']}\n\n"
        "Does this paper genuinely belong to the topic above? "
        "Answer only 'yes' or 'no'."
    )
    return _gemini_generate(prompt).lower().startswith("yes")


def summarize_paper(paper: dict) -> str:
    prompt = (
        f"Title: {paper['title']}\n\n"
        f"Abstract: {paper['abstract']}\n\n"
        "Write a 3-sentence plain-English summary of this paper suitable for a weekly digest. "
        "Focus on what was studied, what was found, and why it matters."
    )
    return _gemini_generate(prompt)


def build_email_html(digest: dict[str, list[dict]]) -> str:
    today = datetime.now().strftime("%B %d, %Y")
    sections = ""
    for topic, papers in digest.items():
        if not papers:
            continue
        items = ""
        for p in papers:
            url = get_paper_url(p)
            venue = p.get("venue") or "Unknown venue"
            link = f'<a href="{url}">{p["title"]}</a>' if url else p["title"]
            items += f"""
            <div style="margin-bottom:20px;">
                <p style="margin:0;font-weight:bold;">{link}</p>
                <p style="margin:2px 0;color:#666;font-size:13px;">{venue}</p>
                <p style="margin:6px 0;">{p["summary"]}</p>
            </div>
            """
        sections += f"""
        <div style="margin-bottom:32px;">
            <h2 style="border-bottom:2px solid #eee;padding-bottom:6px;">{topic}</h2>
            {items}
        </div>
        """
    return f"""
    <html><body style="font-family:sans-serif;max-width:700px;margin:auto;padding:20px;color:#222;">
        <h1 style="color:#1a1a1a;">AI Research Digest</h1>
        <p style="color:#888;">Week of {today} — peer-reviewed papers only</p>
        <hr style="margin:20px 0;">
        {sections}
        <p style="color:#aaa;font-size:12px;margin-top:40px;">
            Sources: Semantic Scholar · Summaries by Claude
        </p>
    </body></html>
    """


ARCHIVE_FILE = os.path.join(os.path.dirname(__file__), "archive.tsv")
ARCHIVE_HEADERS = ["date_sent", "topic", "title", "authors", "venue", "url", "summary"]


def archive_papers(digest: dict[str, list[dict]]):
    date_sent = datetime.now().strftime("%Y-%m-%d")
    file_exists = os.path.isfile(ARCHIVE_FILE)
    with open(ARCHIVE_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ARCHIVE_HEADERS, delimiter="\t")
        if not file_exists:
            writer.writeheader()
        for topic, papers in digest.items():
            for p in papers:
                authors = ", ".join(a.get("name", "") for a in p.get("authors", []))
                writer.writerow({
                    "date_sent": date_sent,
                    "topic": topic,
                    "title": p.get("title", ""),
                    "authors": authors,
                    "venue": p.get("venue") or "",
                    "url": get_paper_url(p),
                    "summary": p.get("summary", ""),
                })


def send_email(subject: str, html_body: str):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = RECIPIENT_EMAIL
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())


def load_recent_titles(runs: int = 3) -> set[str]:
    """Return titles sent in the last `runs` digest runs."""
    if not os.path.isfile(ARCHIVE_FILE):
        return set()
    with open(ARCHIVE_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    dates = sorted({r["date_sent"] for r in rows}, reverse=True)[:runs]
    return {r["title"].lower() for r in rows if r["date_sent"] in dates}


def main():
    all_papers: dict[str, list[dict]] = {}
    recent_titles = load_recent_titles()

    for topic, queries in TOPICS.items():
        topic_papers = []
        for query in queries:
            results = fetch_papers(query, days_back=30)
            topic_papers.extend([p for p in results if is_valid_paper(p)])

        topic_papers = deduplicate(topic_papers)
        topic_papers = [p for p in topic_papers if p.get("title", "").lower() not in recent_titles]
        topic_papers = [p for p in topic_papers if is_relevant(p, topic)]
        topic_papers.sort(key=lambda p: p.get("citationCount") or 0, reverse=True)
        topic_papers = topic_papers[:2]

        for paper in topic_papers:
            paper["summary"] = summarize_paper(paper)

        all_papers[topic] = topic_papers
        print(f"{topic}: {len(topic_papers)} papers")

    total = sum(len(v) for v in all_papers.values())
    if total == 0:
        print("No papers found this week.")
        return

    html = build_email_html(all_papers)
    subject = f"AI Research Digest — {datetime.now().strftime('%B %d, %Y')}"
    archive_papers(all_papers)
    send_email(subject, html)
    print(f"Email sent: {subject}")


def run_scheduler():
    schedule.every().friday.at("09:00").do(main)

    print("Scheduler started. Will send digest every Friday at 9:00 AM.")
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--schedule":
        run_scheduler()
    else:
        main()
