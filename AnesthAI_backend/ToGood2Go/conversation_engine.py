# conversation_engine.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from questions import QUESTIONS


@dataclass
class ConversationState:
    # structured information we collect
    slots: Dict[str, Optional[str]] = field(default_factory=dict)
    # chat history (useful for debugging / later model calls)
    history: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        # initialize all slots to None
        for q in QUESTIONS:
            self.slots.setdefault(q["slot"], None)

    @property
    def unanswered_required_slots(self) -> List[str]:
        """Return list of slot names that are required and still None."""
        missing = []
        for q in QUESTIONS:
            if q["required"] and not self.slots.get(q["slot"]):
                missing.append(q["slot"])
        return missing


class ConversationEngine:
    def __init__(self):
        self.state = ConversationState()
        self.current_question_index = 0

    def next_question(self) -> Optional[str]:
        """
        Decide which question to ask next.
        - Follows the fixed order in QUESTIONS
        - Skips questions whose slot is already filled
        """
        for i in range(self.current_question_index, len(QUESTIONS)):
            q = QUESTIONS[i]
            slot = q["slot"]
            if not self.state.slots.get(slot):
                self.current_question_index = i
                return q["text"]

        # No more questions that need asking
        return None

    def mark_slots_from_llm(self, slot_updates: Dict[str, Optional[str]]):
        """
        Update state.slots using extracted info from the model.
        Only overwrite if model gives a non-empty value.
        """
        for slot, value in slot_updates.items():
            if value:
                self.state.slots[slot] = value

    def is_complete(self) -> bool:
        """Check if all required slots are filled."""
        return len(self.state.unanswered_required_slots) == 0
