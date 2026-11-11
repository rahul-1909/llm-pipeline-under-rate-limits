"""
Deterministic Grader with tolerant pattern matching.
Grades answers according to strict rules:
- correct: key facts present (supporting aliases/formatting variants) AND cited page matches expected page.
- refused_correctly: refusal on an unanswerable question.
- refused_wrongly: refusal on an answerable question.
- wrong: everything else (fact mismatch, page mismatch, hallucination, or error).
"""
import re
from typing import Dict, Any, List, Tuple, Union
from .verifier import is_refusal, extract_citations

class AnswerGrader:
    def _match_fact(self, fact_spec: Union[str, List[str]], text: str) -> bool:
        """
        Matches either a single fact string or any alias in a list of aliases.
        Supports normalized exponent formatting: 1e-9, 10^-9, 10^{-9}, 109.
        """
        aliases = [fact_spec] if isinstance(fact_spec, str) else fact_spec
        text_lower = text.lower()
        norm_text = text_lower.replace(" ", "").replace("{", "").replace("}", "").replace("\\", "")

        for alias in aliases:
            a_lower = alias.lower()
            a_norm = a_lower.replace(" ", "").replace("{", "").replace("}", "").replace("\\", "")
            
            # Direct substring check
            if a_lower in text_lower or a_norm in norm_text:
                return True
            
            # Exponent alias expansion: e.g. 1e-9 vs 10^-9 vs 109
            if any(e in a_lower for e in ["1e-9", "10^-9", "109"]):
                if any(exp in text_lower or exp in norm_text for exp in ["1e-9", "10^-9", "10^{-9}", "109", "10^(-9)"]):
                    return True

        return False

    def grade(
        self,
        question_type: str,
        expected_pages: List[Tuple[str, int]],
        key_facts: List[Union[str, List[str]]],
        answer_text: str,
        status: str = "ok"
    ) -> Dict[str, Any]:
        """
        Returns:
          grade: 'correct' | 'wrong' | 'refused_correctly' | 'refused_wrongly'
          reason: explanation of grading decision
        """
        if status != "ok":
            return {
                "grade": "",
                "reason": f"Request failed with status '{status}'."
            }

        text_is_refusal = is_refusal(answer_text)

        # 1. Unanswerable question evaluation
        if question_type == "unanswerable":
            if text_is_refusal:
                return {
                    "grade": "refused_correctly",
                    "reason": "Correctly refused unanswerable question."
                }
            else:
                return {
                    "grade": "wrong",
                    "reason": "Failed to refuse unanswerable question (hallucinated content)."
                }

        # 2. Answerable question evaluation
        if text_is_refusal:
            return {
                "grade": "refused_wrongly",
                "reason": "Incorrectly refused an answerable question that is present in the papers."
            }

        # 3. Check citations
        citations = extract_citations(answer_text)
        if not citations:
            return {
                "grade": "wrong",
                "reason": "Missing citations in answerable response."
            }

        cited_tuples = {(c["paper_id"], c["page"]) for c in citations if c["paper_id"] and c["page"]}
        
        # Check if at least one expected page was cited
        matched_page = False
        for exp_paper, exp_page in expected_pages:
            if (exp_paper, exp_page) in cited_tuples:
                matched_page = True
                break

        if not matched_page:
            return {
                "grade": "wrong",
                "reason": f"Page mismatch: cited {cited_tuples}, expected one of {expected_pages}."
            }

        # 4. Check key facts with alias tolerance
        missing_facts = []
        for fact_spec in key_facts:
            if not self._match_fact(fact_spec, answer_text):
                missing_facts.append(str(fact_spec))

        if missing_facts:
            return {
                "grade": "wrong",
                "reason": f"Missing key factual claims: {missing_facts}."
            }

        return {
            "grade": "correct",
            "reason": "All key facts present and cited pages match expected ground truth."
        }
