"""
Automated conversation tester for AnæstesiCare.

Simulates a full patient conversation by sending pre-defined responses
to the API. Supports multiple patient personas for different test scenarios.

Usage:
    python test_conversation.py                    # default healthy patient
    python test_conversation.py --persona complex  # patient with complications
    python test_conversation.py --persona chatty   # patient who asks questions
    python test_conversation.py --persona minimal  # terse one-word answers
    python test_conversation.py --base-url http://localhost:8000  # custom API URL

Requirements:
    pip install requests
"""

import argparse
import json
import sys
import time
import requests
from typing import List, Dict, Optional

# ============================================================================
# Patient personas — each maps question IDs to simulated patient responses.
#
# q0 is now kept active until the patient responds, so all personas must
# include a q0 response as their first message.
#
# FREE_CHAT awareness: when the bot asks "Is there anything else on your
# mind, or are you ready to continue?", the tester sends a readiness signal.
# This is handled automatically by the fallback mechanism — responses like
# "No, that's all" or "Ready" will satisfy the FREE_CHAT completion criteria.
# ============================================================================

PERSONAS: Dict[str, Dict[str, List[str]]] = {
    # ------------------------------------------------------------------
    # HEALTHY: straightforward patient, no complications
    # ------------------------------------------------------------------
    "healthy": {
        "name": "Healthy Patient",
        "responses": [
            "I'm ready to begin",                              # q0 — acknowledge greeting
            "My name is Anna Jensen",                          # q1
            "34",                                              # q2
            "No, never",                                       # q3 (no follow-up)
            "No, not that I know of",                          # q4
            "No",                                              # q5
            "No, I don't have sleep apnea",                    # q6
            "No, nothing like that",                           # q7
            "No allergies",                                    # q8
            "No, I don't take any prescription medications",   # q9
            "No blood thinners",                               # q10
            "Just a multivitamin, that's it",                  # q11
            "No heart problems",                               # q12
            "No breathing issues",                             # q13
            "No, I can walk and climb stairs fine",            # q14
            "No reflux or heartburn",                          # q15
            "No, I've been healthy",                           # q16
            "No, I don't smoke",                               # q17
            "Maybe a glass of wine on weekends, so about 2 per week",  # q18
            "No, never",                                       # q19
            "A little nervous but I'm okay",                   # q20
            "No, I think that covers everything",              # q21
            "No additional questions",                         # q22
        ],
    },

    # ------------------------------------------------------------------
    # COMPLEX: patient with multiple conditions and follow-ups triggered
    # ------------------------------------------------------------------
    "complex": {
        "name": "Complex Patient",
        "responses": [
            "Yes, let's get started",                          # q0 — acknowledge greeting
            "Marius Petersen",                                 # q1
            "67",                                              # q2
            "Yes, twice",                                      # q3 → triggers q3b
            "I felt very nauseous after the last time and vomited for hours",  # q3b
            "Yes, my father had a bad reaction",               # q4 → triggers q4b
            "It was malignant hyperthermia",                   # q4b
            "Yes, I was told I have a difficult airway",       # q5
            "Yes, I use a CPAP machine every night for sleep apnea",  # q6
            "I have two dental implants on the upper jaw",     # q7
            "Yes, I have allergies",                           # q8 → triggers q8b
            "Penicillin — I get a rash. And latex gives me hives",  # q8b
            "Yes, I take medications",                         # q9 → triggers q9b
            "Metoprolol 50mg, Lisinopril 10mg, and Metformin 500mg",  # q9b
            "Yes, I take aspirin daily — 81mg",                # q10
            "Yes I take some supplements",                     # q11 → triggers q11b
            "Fish oil, vitamin D, and glucosamine",            # q11b
            "Yes, I had a stent placed two years ago and I have high blood pressure",  # q12
            "I have mild COPD",                                # q13
            "Yes, I get short of breath climbing stairs",      # q14
            "Yes, I take omeprazole for acid reflux",          # q15
            "No recent infections or fever",                   # q16
            "Yes, I smoke",                                    # q17 → triggers q17b
            "About 30 years, half a pack a day",               # q17b
            "About 10 beers a week",                           # q18
            "No, no drugs",                                    # q19
            "Yes, I'm quite worried about the anesthesia given my family history",  # q20
            "I also had a knee replacement 5 years ago, no issues with that anesthesia though",  # q21
            "No further questions",                            # q22
        ],
    },

    # ------------------------------------------------------------------
    # CHATTY: patient who asks questions and adds extra info mid-conversation.
    # Tests FREE_CHAT mode — the bot will ask "Is there anything else on
    # your mind, or are you ready to continue?" and the patient signals ready.
    # ------------------------------------------------------------------
    "chatty": {
        "name": "Chatty Patient",
        "responses": [
            "It's a lovely day today, isn't it?",              # q0 — triggers FREE_CHAT
            "Ready to begin",                                  # q0 FREE_CHAT — signals ready
            "I'm Sarah, nice to meet you!",                    # q1
            "Why do you need my age? I'm 45",                  # q2 — answer + question
            "Yes once, why do you ask?",                       # q3 — answer + question → triggers q3b
            "No problems, it went fine",                       # q3b
            "What do you mean by adverse reaction? No I don't think so",  # q4 — question + answer
            "Will I be intubated? No, nobody told me that",    # q5 — question + answer
            "What is sleep apnea exactly? No I don't have it", # q6 — question + answer
            "No dental issues",                                # q7
            "No allergies at all",                             # q8
            "No medications",                                  # q9
            "What are blood thinners for? No I don't take any",  # q10 — question + answer
            "Just vitamin C sometimes",                        # q11
            "No heart problems, should I be worried?",         # q12 — answer + question
            "No, that's all",                                  # q12 FREE_CHAT — signals ready
            "No breathing problems",                           # q13
            "No shortness of breath",                          # q14
            "No reflux",                                       # q15
            "No recent health changes",                        # q16
            "No I don't smoke",                                # q17
            "I don't drink alcohol",                           # q18
            "No drugs",                                        # q19
            "I'm a bit nervous, is that normal?",              # q20 — question
            "Nothing else to add",                             # q21
            "No additional questions",                         # q22
        ],
    },

    # ------------------------------------------------------------------
    # MINIMAL: very short/terse answers to test edge cases
    # ------------------------------------------------------------------
    "minimal": {
        "name": "Minimal Patient",
        "responses": [
            "Ready",                   # q0 — acknowledge greeting
            "Erik",                    # q1
            "29",                      # q2
            "No",                      # q3
            "No",                      # q4
            "No",                      # q5
            "No",                      # q6
            "No",                      # q7
            "None",                    # q8
            "No",                      # q9
            "No",                      # q10
            "No",                      # q11
            "No",                      # q12
            "No",                      # q13
            "No",                      # q14
            "No",                      # q15
            "No",                      # q16
            "No",                      # q17
            "None",                    # q18
            "No",                      # q19
            "No",                      # q20
            "Nothing",                 # q21
            "No",                      # q22
        ],
    },
}


# ============================================================================
# Readiness signals — used as fallback responses when the bot enters
# FREE_CHAT and asks "Is there anything else on your mind, or are you
# ready to continue?"
# ============================================================================

FREE_CHAT_FALLBACKS = [
    "No, that's all",
    "I'm ready to continue",
    "Nothing else",
    "Yes, let's continue",
    "No more questions",
]

GENERIC_FALLBACKS = [
    "Yes",
    "No, nothing else",
    "That's correct",
    "Yes, that's right",
]


# ============================================================================
# Colors for terminal output
# ============================================================================

class C:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


# ============================================================================
# Test runner
# ============================================================================

class ConversationTester:
    def __init__(self, base_url: str, persona_key: str, delay: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.api = f"{self.base_url}/api/v1"
        self.persona = PERSONAS[persona_key]
        self.delay = delay
        self.conversation_id = f"test-{persona_key}-{int(time.time())}"
        self.turn_count = 0
        self.free_chat_turns = 0
        self.errors: List[str] = []

    def run(self) -> bool:
        """Run the full conversation. Returns True if completed successfully."""
        print(f"\n{C.BOLD}{'='*60}")
        print(f"  Testing persona: {self.persona['name']}")
        print(f"  Conversation ID: {self.conversation_id}")
        print(f"  API: {self.api}")
        print(f"{'='*60}{C.RESET}\n")

        # Step 1: Start conversation
        try:
            bot_text, done = self._start()
            self._print_bot(bot_text)
            if done:
                self._error("Conversation ended immediately after start")
                return self._report()
        except Exception as e:
            self._error(f"Failed to start conversation: {e}")
            return self._report()

        # Step 2: Send each prepared response
        response_iter = iter(self.persona["responses"])
        done = False

        for response in response_iter:
            if done:
                print(f"\n{C.YELLOW}⚠ Conversation ended before all responses were used{C.RESET}")
                break

            if self.delay > 0:
                time.sleep(self.delay)

            self.turn_count += 1
            self._print_patient(response)

            try:
                bot_text, done = self._send_message(response)
                self._print_bot(bot_text)
            except Exception as e:
                self._error(f"Turn {self.turn_count} failed: {e}")
                return self._report()

            # If bot entered FREE_CHAT and we have no more prepared responses
            # that handle it, send a readiness signal automatically
            if not done and self._looks_like_free_chat(bot_text):
                self.free_chat_turns += 1
                if self.free_chat_turns > 5:
                    self._error("Stuck in FREE_CHAT for more than 5 consecutive turns")
                    return self._report()

                fallback = FREE_CHAT_FALLBACKS[
                    (self.free_chat_turns - 1) % len(FREE_CHAT_FALLBACKS)
                ]
                self.turn_count += 1
                self._print_patient(f"{fallback} (auto free-chat response)")
                try:
                    bot_text, done = self._send_message(fallback)
                    self._print_bot(bot_text)
                except Exception as e:
                    self._error(f"FREE_CHAT fallback turn failed: {e}")
                    return self._report()
            else:
                self.free_chat_turns = 0

        # Step 3: If not done yet, try generic fallbacks
        if not done:
            print(f"\n{C.YELLOW}⚠ Responses exhausted, trying generic fallbacks...{C.RESET}")
            for fallback in GENERIC_FALLBACKS:
                if done:
                    break
                self.turn_count += 1
                self._print_patient(f"{fallback} (generic fallback)")
                try:
                    bot_text, done = self._send_message(fallback)
                    self._print_bot(bot_text)
                except Exception as e:
                    self._error(f"Generic fallback turn failed: {e}")
                    break

        if not done:
            self._error("Conversation did not reach 'done' state")

        # Step 4: Fetch summary if done
        if done:
            try:
                summary = self._get_summary()
                self._print_summary(summary)
            except Exception as e:
                self._error(f"Failed to fetch summary: {e}")

        # Step 5: Check state endpoint
        try:
            state = self._get_state()
            print(f"\n{C.GRAY}Final state: {json.dumps(state, indent=2)}{C.RESET}")
        except Exception as e:
            self._error(f"State endpoint failed: {e}")

        # Step 6: Cleanup
        try:
            self._delete_conversation()
        except Exception:
            pass

        return self._report()

    def _looks_like_free_chat(self, bot_text: str) -> bool:
        """Detect if the bot's reply ended with the FREE_CHAT question."""
        return "anything else on your mind" in bot_text.lower() or \
               "ready to continue" in bot_text.lower()

    # ------------------------------------------------------------------
    # API calls
    # ------------------------------------------------------------------

    def _start(self):
        r = requests.post(
            f"{self.api}/conversations/start",
            json={"conversation_id": self.conversation_id},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        return data["bot_text"], data.get("done", False)

    def _send_message(self, message: str):
        r = requests.post(
            f"{self.api}/conversations/{self.conversation_id}/message",
            json={"message": message},
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        return data["bot_text"], data.get("done", False)

    def _get_state(self):
        r = requests.get(
            f"{self.api}/conversations/{self.conversation_id}/state",
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    def _get_summary(self):
        r = requests.get(
            f"{self.api}/conversations/{self.conversation_id}/summary",
            timeout=120,
        )
        r.raise_for_status()
        return r.json()

    def _delete_conversation(self):
        requests.delete(
            f"{self.api}/conversations/{self.conversation_id}",
            timeout=10,
        )

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _print_bot(self, text: str):
        print(f"  {C.BLUE}🤖 Bot:{C.RESET} {text}\n")

    def _print_patient(self, text: str):
        print(f"  {C.GREEN}👤 Patient (turn {self.turn_count}):{C.RESET} {text}")

    def _print_summary(self, summary: dict):
        print(f"\n{C.BOLD}{'─'*60}")
        print(f"  SUMMARY")
        print(f"{'─'*60}{C.RESET}")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    def _error(self, msg: str):
        self.errors.append(msg)
        print(f"  {C.RED}✗ {msg}{C.RESET}")

    def _report(self) -> bool:
        print(f"\n{C.BOLD}{'='*60}")
        print(f"  RESULTS")
        print(f"{'='*60}{C.RESET}")
        print(f"  Turns:       {self.turn_count}")
        print(f"  Free chat:   {self.free_chat_turns} auto-handled turn(s)")
        print(f"  Errors:      {len(self.errors)}")

        if self.errors:
            for e in self.errors:
                print(f"  {C.RED}✗ {e}{C.RESET}")
            print(f"\n  {C.RED}{C.BOLD}FAILED{C.RESET}\n")
            return False
        else:
            print(f"\n  {C.GREEN}{C.BOLD}PASSED{C.RESET}\n")
            return True


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Automated conversation tester")
    parser.add_argument(
        "--persona", "-p",
        choices=list(PERSONAS.keys()) + ["all"],
        default="healthy",
        help="Patient persona to simulate (default: healthy)",
    )
    parser.add_argument(
        "--base-url", "-u",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=0.0,
        help="Delay in seconds between messages (default: 0)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available personas and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable personas:")
        for key, persona in PERSONAS.items():
            print(f"  {key:12s} — {persona['name']} ({len(persona['responses'])} responses)")
        print()
        sys.exit(0)

    personas_to_run = list(PERSONAS.keys()) if args.persona == "all" else [args.persona]
    results = {}

    for persona_key in personas_to_run:
        tester = ConversationTester(
            base_url=args.base_url,
            persona_key=persona_key,
            delay=args.delay,
        )
        results[persona_key] = tester.run()

    if len(results) > 1:
        print(f"\n{C.BOLD}{'='*60}")
        print(f"  ALL RESULTS")
        print(f"{'='*60}{C.RESET}")
        for key, passed in results.items():
            status = f"{C.GREEN}PASSED{C.RESET}" if passed else f"{C.RED}FAILED{C.RESET}"
            print(f"  {key:12s} — {status}")
        print()

    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()