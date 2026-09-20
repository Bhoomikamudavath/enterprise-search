import xml.etree.ElementTree as ET
import html
import re
import json
from pathlib import Path

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "Posts.xml"
OUTPUT_FILE = DATA_DIR / "documents.jsonl"


def clean_html(raw_html):
    text = html.unescape(raw_html)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_posts(xml_path):
    questions = {}
    answers_by_parent = {}

    for event, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag != "row":
            continue

        post_type = elem.get("PostTypeId")
        post_id = elem.get("Id")

        if post_type == "1":
            questions[post_id] = {
                "id": post_id,
                "title": elem.get("Title", ""),
                "body": clean_html(elem.get("Body", "")),
                "tags": [t for t in elem.get("Tags", "").split("|") if t],
                "accepted_answer_id": elem.get("AcceptedAnswerId"),
                "score": int(elem.get("Score", 0)),
            }
        elif post_type == "2":
            parent_id = elem.get("ParentId")
            answer = {
                "id": post_id,
                "body": clean_html(elem.get("Body", "")),
                "score": int(elem.get("Score", 0)),
            }
            answers_by_parent.setdefault(parent_id, []).append(answer)

        elem.clear()

    return questions, answers_by_parent


def pick_best_answer(question, answers):
    if not answers:
        return None
    accepted_id = question.get("accepted_answer_id")
    for a in answers:
        if a["id"] == accepted_id:
            return a
    return max(answers, key=lambda a: a["score"])


def build_documents(questions, answers_by_parent):
    docs = []
    for qid, q in questions.items():
        answers = answers_by_parent.get(qid, [])
        best_answer = pick_best_answer(q, answers)

        doc = {
            "doc_id": qid,
            "title": q["title"],
            "question_body": q["body"],
            "answer_body": best_answer["body"] if best_answer else "",
            "tags": q["tags"],
            "score": q["score"],
            "url": f"https://ai.stackexchange.com/questions/{qid}",
        }
        doc["full_text"] = f"{doc['title']} {doc['question_body']} {doc['answer_body']}".strip()
        docs.append(doc)
    return docs


def main():
    print("Parsing XML...")
    questions, answers_by_parent = parse_posts(INPUT_FILE)
    print(f"Found {len(questions)} questions, {sum(len(v) for v in answers_by_parent.values())} answers")

    docs = build_documents(questions, answers_by_parent)
    docs = [d for d in docs if d["full_text"]]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")

    print(f"Wrote {len(docs)} documents to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()