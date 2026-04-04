import os
import smtplib
import requests
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from dotenv import load_dotenv
from google import genai
import schedule
import time

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
_gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GMAIL_ADDRESS = os.environ["GMAIL_ADDRESS"]
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
RECIPIENT_EMAIL = os.environ["RECIPIENT_EMAIL"]

TOPICS = {
    "AI for Education": [
        "AI education", "LLM tutoring", "adaptive learning AI",
        "artificial intelligence classroom", "AI teaching"
    ],
    "AI for Health": [
        "AI healthcare", "clinical AI", "medical LLM",
        "artificial intelligence diagnosis", "AI radiology"
    ],
    "Model Behavior": [
        "LLM alignment", "model behavior", "AI safety",
        "large language model evaluation", "AI robustness"
    ],
    "AI Ethics": [
        "AI ethics", "algorithmic bias", "AI fairness",
        "responsible AI", "AI accountability"
    ],
    "Breaking News in Tech": [
        "AI breakthrough", "emerging AI technology", "tech innovation",
        "artificial intelligence advances", "machine learning breakthrough",
        "new AI model", "AI research breakthrough"
    ],
    "Major AI Players": [
        "OpenAI research", "Google AI", "Anthropic AI", "Meta AI",
        "Microsoft AI", "xAI research", "DeepMind AI", "AI company research"
    ],
}

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def fetch_papers(query: str, days_back: int = 7, limit: int = 5) -> list[dict]:
    since = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    params = {
        "query": query,
        "fields": "title,authors,year,publicationDate,publicationTypes,venue,externalIds,abstract,url",
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


def summarize_paper(paper: dict) -> str:
    prompt = (
        f"Title: {paper['title']}\n\n"
        f"Abstract: {paper['abstract']}\n\n"
        "Write a 3-sentence plain-English summary of this paper suitable for a weekly digest. "
        "Focus on what was studied, what was found, and why it matters."
    )
    response = _gemini_client.models.generate_content(
        model="gemini-2.0-flash-lite", contents=prompt
    )
    return response.text.strip()


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


def send_email(subject: str, html_body: str):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = RECIPIENT_EMAIL
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())


def main():
    all_papers: dict[str, list[dict]] = {}

    for topic, queries in TOPICS.items():
        topic_papers = []
        for query in queries:
            results = fetch_papers(query)
            topic_papers.extend([p for p in results if is_valid_paper(p)])

        topic_papers = deduplicate(topic_papers)[:2]

        for paper in topic_papers:
            paper["summary"] = summarize_paper(paper)
            time.sleep(13)  # stay under 5 req/min free tier limit

        all_papers[topic] = topic_papers
        print(f"{topic}: {len(topic_papers)} papers")

    total = sum(len(v) for v in all_papers.values())
    if total == 0:
        print("No papers found this week.")
        return

    html = build_email_html(all_papers)
    subject = f"AI Research Digest — {datetime.now().strftime('%B %d, %Y')}"
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
