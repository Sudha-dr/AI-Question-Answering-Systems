import re


KNOWLEDGE_BASE = [
    (
        "What is artificial intelligence?",
        "Artificial intelligence (AI) is the ability of a computer system to perform tasks that usually need human intelligence, such as understanding language or recognizing patterns.",
    ),
    (
        "What is machine learning?",
        "Machine learning is a type of AI. It lets computers learn patterns from examples instead of following only hand-written rules.",
    ),
    (
        "What is natural language processing?",
        "Natural language processing (NLP) is a field of AI that helps computers work with human language, including reading, understanding, and generating text.",
    ),
    (
        "What is a question answering system?",
        "A question answering system takes a question and returns an answer from information it has been given or can access.",
    ),
    (
        "How does this demo find an answer?",
        "This demo compares the important words in your question with the questions in its knowledge base, then returns the answer for the closest match.",
    ),
    (
        "How can I add more answers?",
        "Add another question and answer pair to the KNOWLEDGE_BASE list in app.py, save the file, and run the program again.",
    ),
]

STOP_WORDS = {
    "a", "an", "and", "are", "can", "do", "does", "explain", "for", "how",
    "i", "is", "it", "me", "of", "please", "tell", "that", "the", "this",
    "to", " was", "what", "when", "where", "which", "who", "why", "with",
}


def important_words(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if word not in STOP_WORDS}


def answer_question(question):
    question_words = important_words(question)
    if not question_words:
        return "Please ask a question with a few topic words."

    best_answer = None
    best_score = 0

    for known_question, answer in KNOWLEDGE_BASE:
        known_words = important_words(known_question)
        score = len(question_words & known_words) / len(question_words)
        if score > best_score:
            best_score = score
            best_answer = answer

    if best_score < 0.25:
        return "I do not know that yet. Add information about it to KNOWLEDGE_BASE in app.py."

    return best_answer


def main():
    print("Simple Question Answering System")
    print("Ask a question, or type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not question:
            print("Please type a question.\n")
            continue

        print(f"Answer: {answer_question(question)}\n")


if __name__ == "__main__":
    main()
