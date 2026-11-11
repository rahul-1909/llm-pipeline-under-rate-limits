"""
Test script to demonstrate the deterministic grader across 5 test cases:
1. correct
2. refused_correctly
3. wrong (page mismatch / missing facts)
4. refused_wrongly (refusing an answerable question)
5. wrong (hallucinating an answer to an unanswerable question)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.grader import AnswerGrader

grader = AnswerGrader()

examples = [
    {
        "name": "Example 1: Ground-truth correct answer (Q1)",
        "question_type": "answerable",
        "expected_pages": [("1706.03762v7", 7)],
        "key_facts": ["0.9", "0.98", "109", "4000"],
        "answer_text": "The Transformer uses the Adam optimizer with beta1 = 0.9, beta2 = 0.98, epsilon = 109, and 4000 warmup steps [1706.03762v7, Page 7].",
        "status": "ok"
    },
    {
        "name": "Example 2: Legitimate refusal on unanswerable question (Q5 ImageNet)",
        "question_type": "unanswerable",
        "expected_pages": [],
        "key_facts": [],
        "answer_text": "The provided library papers ('Attention Is All You Need', 'Moshi', 'Dynamic Chunking') do not contain information about ImageNet top-1 accuracy.",
        "status": "ok"
    },
    {
        "name": "Example 3: Wrong (Cited wrong page / hallucinated page)",
        "question_type": "answerable",
        "expected_pages": [("1706.03762v7", 7)],
        "key_facts": ["0.9", "0.98"],
        "answer_text": "The Adam optimizer settings are beta1 = 0.9 and beta2 = 0.98 [1706.03762v7, Page 14].",
        "status": "ok"
    },
    {
        "name": "Example 4: Refused Wrongly (Pipeline refused an answerable question)",
        "question_type": "answerable",
        "expected_pages": [("1706.03762v7", 7)],
        "key_facts": ["0.9", "0.98"],
        "answer_text": "The library papers do not contain sufficient information to answer this question.",
        "status": "ok"
    },
    {
        "name": "Example 5: Wrong (Hallucinated answer on unanswerable question)",
        "question_type": "unanswerable",
        "expected_pages": [],
        "key_facts": [],
        "answer_text": "The Transformer achieves an ImageNet top-1 accuracy of 82.5% [1706.03762v7, Page 5].",
        "status": "ok"
    }
]

def run_tests():
    print("=== Demonstrating 5 Grader Examples ===")
    for ex in examples:
        res = grader.grade(
            question_type=ex["question_type"],
            expected_pages=ex["expected_pages"],
            key_facts=ex["key_facts"],
            answer_text=ex["answer_text"],
            status=ex["status"]
        )
        print(f"\n{ex['name']}:")
        print(f"  Result Grade: {res['grade']}")
        print(f"  Reason:       {res['reason']}")

if __name__ == "__main__":
    run_tests()
