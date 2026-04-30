"""
Automated conversation tester for AnæstesiCare — Evaluation Edition.

Runs patient personas against the chatbot API with question-aware routing:
the API returns `current_question_id` after each message, and the script
uses it to serve the correct persona response — no guessing, no drift.

Usage:
    python test_conversation.py                              # default: uncertain x1
    python test_conversation.py -p uncertain -n 10           # 10 runs of uncertain
    python test_conversation.py -p all -n 10                 # 10 runs of each persona
    python test_conversation.py -p contradictory -n 5 -d 1   # 5 runs, 1s delay
    python test_conversation.py --list                       # show all personas

Transcripts are saved to a 'transcripts/' folder next to this script.
Filenames: {persona}__run{N}__{timestamp}.txt

Requirements:
    pip install requests

API requirement:
    The /message and /state endpoints must return `current_question_id`.
"""

import argparse
import json
import os
import sys
import time
import requests
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# ============================================================================
# Transcript output directory — created next to this script
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPT_DIR = os.path.join(SCRIPT_DIR, "transcripts")


# ============================================================================
# Patient personas — keyed by question ID
#
# Each value is a list of responses. The first is used on the first ask,
# the second if the bot re-asks, etc. If all are exhausted, the last
# one is repeated.
# ============================================================================

PERSONAS: Dict[str, Dict] = {

    # ------------------------------------------------------------------
    # UNCERTAIN: vague, unsure answers — tests completion criteria
    # Based on 8/20 real participants who used "I think so", "not sure",
    # "I don't remember the name", etc.
    # ------------------------------------------------------------------
    "uncertain": {
        "name": "Uncertain Patient",
        "responses": {
            "q0":   ["I think im ready"],
            "q1":   ["Lars"],
            "q2":   ["im 52 i think, wait yes 52"],
            "q3":   ["I think so, maybe once but im not sure"],
            "q3b":  ["I dont really remember, it was a long time ago"],
            "q4":   ["I dont know, noone has ever told me about that"],
            "q4b":  ["im not sure what it was exactly"],
            "q5":   ["not that i know of"],
            "q6":   ["im not sure, I snore alot but ive never been diagnosed"],
            "q7":   ["no I dont think so"],
            "q8":   ["not that I know of, I dont take drugs that often"],
            "q8b":  ["I think maybe some rash once but im not sure from what"],
            "q9":   ["I take some pills but I dont remember what they are called"],
            "q9b":  ["something for my blood pressure I think, and maybe something else"],
            "q10":  ["I think I might take aspirin but im not sure"],
            "q11":  ["no i dont think so"],
            "q11b": ["not sure, maybe some vitamins"],
            "q12":  ["I dont think so, nothing serious at least"],
            "q13":  ["not really, maybe sometimes"],
            "q14":  ["I get a bit winded sometimes but I think thats normal"],
            "q15":  ["maybe, I dont know what acid reflux is exactly"],
            "q16":  ["not that I can think of"],
            "q17":  ["I used to smoke but I quit, well mostly quit"],
            "q17b": ["I dont know, like a year ago, and now just sometimes at parties"],
            "q18":  ["not much, a few glasses here and there"],
            "q19":  ["no, never"],
            "q19b": ["no never"],
            "q20":  ["im a bit nervous I suppose but its fine"],
            "q21":  ["I cant think of anything but I might be forgetting something"],
            "q22":  ["no I think we covered everything"],
        },
    },

    # ------------------------------------------------------------------
    # INQUISITIVE: answers + questions in same message
    # Based on 6/20 real participants who asked "should I be worried?",
    # "what is COPD?", "what does that mean for me?" etc.
    # ------------------------------------------------------------------
    "inquisitive": {
        "name": "Inquisitive Patient",
        "responses": {
            "q0":   ["Sure, but first can you tell me how long this will take?"],
            "q1":   ["My name is Sofia, nice to meet you"],
            "q2":   ["I'm 38, is that relevant for the anesthesia?"],
            "q3":   ["Yes once when I was a child, what type of anesthesia will I get?"],
            "q3b":  ["No problems but I was dizzy after, is that normal?"],
            "q4":   ["My uncle had some reaction but I dont know the details, should I be worried?"],
            "q4b":  ["I think it was something with his muscles, what does that mean for me?"],
            "q5":   ["No, but what does intubation actually involve?"],
            "q6":   ["No sleep apnea, but what does it mean if you have it during surgery?"],
            "q7":   ["I have a retainer wire on the back of my teeth from braces, does that matter?"],
            "q8":   ["I'm allergic to latex, is that a problem for the operation?"],
            "q8b":  ["I get a rash from latex gloves, will you use latex during the procedure?"],
            "q9":   ["Yes, I take blood pressure medication and something for my thyroid, is that a concern?"],
            "q9b":  ["Losartan for blood pressure and levothyroxine for thyroid"],
            "q10":  ["No blood thinners, but what would happen if I did take them?"],
            "q11":  ["I take fish oil and vitamin D, should I stop before the surgery?"],
            "q11b": ["fish oil and vitamin D, thats all"],
            "q12":  ["No heart issues, but my father had a heart attack at 55, is that relevant?"],
            "q13":  ["No, but I sometimes feel like I cant take a deep breath, is that asthma?"],
            "q14":  ["Only when I run up many flights of stairs, is that concerning?"],
            "q15":  ["Sometimes after spicy food, do I need to fast before surgery?"],
            "q16":  ["I had a cold about 3 weeks ago but im fine now, will that affect anything?"],
            "q17":  ["No I dont smoke, but my husband smokes a lot, does secondhand smoke matter?"],
            "q17b": ["I never smoked myself"],
            "q18":  ["A glass of wine with dinner most nights so maybe 5-6 per week, is that too much?"],
            "q19":  ["No never, but is it true that some drugs interact badly with anesthesia?"],
            "q19b": ["I said no, I dont use any"],
            "q20":  ["Yes honestly im quite nervous, can you tell me what will happen during the procedure?"],
            "q21":  ["I had my appendix out 10 years ago with no problems, should I mention that?"],
            "q22":  ["Actually yes, what happens if something goes wrong during anesthesia?"],
        },
    },

    # ------------------------------------------------------------------
    # CONTRADICTORY: self-corrections, accidental yes/no
    # Based on 6/20 real participants who changed answers, said "yes"
    # then "actually no", or corrected the bot.
    # ------------------------------------------------------------------
    "contradictory": {
        "name": "Contradictory Patient",
        "responses": {
            "q0":   ["yes go"],
            "q1":   ["Peter Nielsen"],
            "q2":   ["Im 47, oh wait no im 48 I had a birthday last week"],
            "q3":   ["yes"],
            "q3b":  [
                "actually wait no I havent, I was thinking of something else sorry",
                "no I really havent had anesthesia before",
            ],
            "q4":   ["yes"],
            "q4b":  [
                "actually im not sure, I might be confusing it with something else",
                "I honestly dont know, sorry",
            ],
            "q5":   ["I think so, well actually no, nobody told me that"],
            "q6":   ["no wait actually I do snore alot and my wife says I stop breathing"],
            "q7":   ["no"],
            "q8":   ["yes"],
            "q8b":  [
                "well actually I dont think its a real allergy, I just get itchy sometimes from plasters",
                "just the plasters, nothing else",
            ],
            "q9":   ["Not anymore, I used to take something but I stopped"],
            "q9b":  ["I stopped taking everything, nothing currently"],
            "q10":  ["No, well actually I take an aspirin every morning but I didnt think that counted"],
            "q11":  ["no, oh wait I do take vitamin D in the winter, does that count?"],
            "q11b": ["just vitamin D, thats it"],
            "q12":  [
                "yes",
                "wait no I dont, I was thinking of my father, hes the one with heart problems",
            ],
            "q13":  ["no"],
            "q14":  ["yes, well sometimes, it depends on the day"],
            "q15":  ["no, well actually I do get heartburn sometimes but I wouldnt call it acid reflux"],
            "q16":  ["no I dont think so, oh actually I had a fever last week but it went away"],
            "q17":  ["no I quit 5 years ago"],
            "q17b": ["I quit 5 years ago, used to smoke about 10 a day for 15 years"],
            "q18":  ["not much, maybe 8-10 beers on the weekend"],
            "q19":  ["no"],
            "q19b": ["no, nothing"],
            "q20":  ["not really, well maybe a little bit about the anesthesia part"],
            "q21":  ["I had knee surgery 3 years ago, I forgot to mention that earlier"],
            "q22":  ["no I think thats everything, oh wait actually can the anesthesiologist see all this?"],
        },
    },
}


# ============================================================================
# Fallback responses
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
    "I think so",
    "No",
    "Nothing more",
    "That's all",
]


# ============================================================================
# Colors
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
    def __init__(
        self,
        base_url: str,
        persona_key: str,
        run_number: int = 1,
        delay: float = 0.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api = f"{self.base_url}/api/v1"
        self.persona_key = persona_key
        self.persona = PERSONAS[persona_key]
        self.run_number = run_number
        self.delay = delay
        self.conversation_id = f"eval-{persona_key}-run{run_number}-{int(time.time())}"
        self.turn_count = 0
        self.free_chat_turns = 0
        self.fallback_count = 0
        self.errors: List[str] = []

        # Transcript log — (role, message, meta)
        self.transcript: List[Tuple[str, str, str]] = []

        # Track response cursor per question ID
        self._response_cursors: Dict[str, int] = {}

        # Track how many patient turns each question required
        self._question_attempts: Dict[str, int] = {}

    def _get_response(self, question_id: str) -> Optional[str]:
        """Get the next prepared response for a question ID."""
        responses = self.persona["responses"].get(question_id)
        if not responses:
            return None

        cursor = self._response_cursors.get(question_id, 0)
        if cursor >= len(responses):
            return responses[-1]  # Repeat last response if re-asked

        self._response_cursors[question_id] = cursor + 1
        return responses[cursor]

    def _is_free_chat(self, bot_text: str) -> bool:
        lower = bot_text.lower()
        return "anything else on your mind" in lower or "ready to continue" in lower

    def run(self) -> Tuple[bool, Dict]:
        """Run the full conversation. Returns (passed, stats)."""
        self._start_time = time.time()

        print(f"\n{C.BOLD}{'='*60}")
        print(f"  Persona: {self.persona['name']}  |  Run {self.run_number}")
        print(f"  Conversation ID: {self.conversation_id}")
        print(f"  API: {self.api}")
        print(f"{'='*60}{C.RESET}\n")

        # Step 1: Start conversation
        try:
            bot_text, done, current_qid = self._start()
            self._log("Assistant", bot_text, f"asking: {current_qid}")
            self._print_bot(bot_text, current_qid)
            if done:
                self._error("Conversation ended immediately after start")
                return self._finalize()
        except Exception as e:
            self._error(f"Failed to start conversation: {e}")
            return self._finalize()

        # Step 2: Respond loop — driven by current_question_id from the API
        max_turns = 60

        while not done and self.turn_count < max_turns:
            if self.delay > 0:
                time.sleep(self.delay)

            # Handle FREE_CHAT (detected from bot text since it's not a real question ID)
            if self._is_free_chat(bot_text):
                self.free_chat_turns += 1
                if self.free_chat_turns > 5:
                    self._error("Stuck in FREE_CHAT for more than 5 consecutive turns")
                    return self._finalize()
                response = FREE_CHAT_FALLBACKS[
                    (self.free_chat_turns - 1) % len(FREE_CHAT_FALLBACKS)
                ]
                meta = f"→ free_chat (auto #{self.free_chat_turns})"
            else:
                self.free_chat_turns = 0

                # Track attempts per question
                if current_qid:
                    self._question_attempts[current_qid] = self._question_attempts.get(current_qid, 0) + 1

                attempts = self._question_attempts.get(current_qid, 0)

                # Loop breaker: if stuck on the same question for too many turns,
                # force a bare answer the bot can't misinterpret
                if attempts > 3:
                    loop_fallbacks = ["Yes", "No", "I don't know", "No", "Yes"]
                    response = loop_fallbacks[(attempts - 4) % len(loop_fallbacks)]
                    meta = f"→ {current_qid} (LOOP BREAKER attempt #{attempts})"
                    print(f"  {C.RED}⚠ Loop detected on {current_qid} ({attempts} attempts) — forcing bare answer{C.RESET}")
                else:
                    # Normal: use the persona's prepared response
                    response = self._get_response(current_qid) if current_qid else None
                    meta = f"→ {current_qid}"

                if response is None:
                    # No prepared response for this question — use fallback
                    self.fallback_count += 1
                    response = GENERIC_FALLBACKS[
                        (self.fallback_count - 1) % len(GENERIC_FALLBACKS)
                    ]
                    meta = f"→ {current_qid} (NO PREPARED RESPONSE — fallback #{self.fallback_count})"
                    print(f"  {C.YELLOW}⚠ No response for {current_qid}, using fallback{C.RESET}")

                    if self.fallback_count > 10:
                        self._error("Too many fallback responses — aborting")
                        return self._finalize()

            self.turn_count += 1
            self._log("Patient", response, meta)
            self._print_patient(response, meta)

            try:
                bot_text, done, current_qid = self._send_message(response)
                bot_meta = f"asking: {current_qid}" if current_qid else "done"
                self._log("Assistant", bot_text, bot_meta)
                self._print_bot(bot_text, current_qid)
            except Exception as e:
                self._error(f"Turn {self.turn_count} failed: {e}")
                return self._finalize()

        if not done and self.turn_count >= max_turns:
            self._error(f"Hit max turn limit ({max_turns})")

        # Step 3: Fetch summary if done
        summary = None
        if done:
            try:
                summary = self._get_summary()
                self._print_summary(summary)
            except Exception as e:
                self._error(f"Failed to fetch summary: {e}")

        # Step 4: Save transcript
        self._save_transcript(summary)

        # Step 5: Cleanup
        try:
            self._delete_conversation()
        except Exception:
            pass

        return self._finalize()
    # API calls
    # ------------------------------------------------------------------

    def _start(self) -> Tuple[str, bool, Optional[str]]:
        r = requests.post(
            f"{self.api}/conversations/start",
            json={"conversation_id": self.conversation_id},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()

        # After start, get the current question ID from state
        state = self._get_state()
        current_qid = state.get("current_question_id")

        return data["bot_text"], data.get("done", False), current_qid

    def _send_message(self, message: str) -> Tuple[str, bool, Optional[str]]:
        r = requests.post(
            f"{self.api}/conversations/{self.conversation_id}/message",
            json={"message": message},
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        return (
            data["bot_text"],
            data.get("done", False),
            data.get("current_question_id"),
        )

    def _get_state(self) -> dict:
        r = requests.get(
            f"{self.api}/conversations/{self.conversation_id}/state",
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    def _get_summary(self) -> dict:
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
    # Transcript saving
    # ------------------------------------------------------------------

    def _save_transcript(self, summary: Optional[dict] = None):
        os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.persona_key}__run{self.run_number}__{timestamp}.txt"
        filepath = os.path.join(TRANSCRIPT_DIR, filename)

        lines = []
        lines.append(f"{self.persona['name']} — Run {self.run_number}")
        lines.append(f"Conversation ID: {self.conversation_id}")
        lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("─" * 50)
        lines.append("")

        for role, message, meta in self.transcript:
            meta_str = f"  [{meta}]" if meta else ""
            lines.append(f"[{role}]{meta_str}: {message}")
            lines.append("")

        lines.append("─" * 50)
        lines.append(f"Total patient turns: {self.turn_count}")
        lines.append(f"Free chat auto-responses: {self.free_chat_turns}")
        lines.append(f"Fallback responses: {self.fallback_count}")
        elapsed = time.time() - self._start_time
        m, s = divmod(int(elapsed), 60)
        lines.append(f"Duration: {m}m {s}s" if m else f"Duration: {s}s")

        # Per-question stats
        questions_asked = {q: a for q, a in self._question_attempts.items()}
        total_questions = len(questions_asked)
        first_attempt = sum(1 for a in questions_asked.values() if a == 1)
        re_asked = {q: a for q, a in questions_asked.items() if a > 1}
        first_attempt_rate = (first_attempt / total_questions * 100) if total_questions > 0 else 0

        lines.append(f"Questions asked: {total_questions}")
        lines.append(f"First-attempt acceptance: {first_attempt}/{total_questions} ({first_attempt_rate:.0f}%)")

        if re_asked:
            lines.append(f"Re-asked questions:")
            for qid, attempts in re_asked.items():
                lines.append(f"  {qid}: {attempts} attempts")

        lines.append("")
        lines.append("Turns per question:")
        for qid, attempts in sorted(questions_asked.items()):
            marker = " ⚠" if attempts > 1 else ""
            lines.append(f"  {qid}: {attempts}{marker}")

        lines.append("")
        lines.append(f"Errors: {len(self.errors)}")
        if self.errors:
            for e in self.errors:
                lines.append(f"  ERROR: {e}")
        lines.append(f"Result: {'PASSED' if not self.errors else 'FAILED'}")

        if summary:
            lines.append("")
            lines.append("─" * 50)
            lines.append("SUMMARY")
            lines.append("─" * 50)
            lines.append(json.dumps(summary, indent=2, ensure_ascii=False))

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"\n  {C.GRAY}📄 Transcript saved: {filepath}{C.RESET}")

    # ------------------------------------------------------------------
    # Logging & output
    # ------------------------------------------------------------------

    def _log(self, role: str, message: str, meta: str = ""):
        self.transcript.append((role, message, meta))

    def _print_bot(self, text: str, qid: Optional[str] = None):
        qid_tag = f" {C.GRAY}[{qid}]{C.RESET}" if qid else ""
        print(f"  {C.BLUE}🤖 Bot{qid_tag}:{C.RESET} {text}\n")

    def _print_patient(self, text: str, meta: str = ""):
        meta_tag = f" {C.GRAY}[{meta}]{C.RESET}" if meta else ""
        print(f"  {C.GREEN}👤 Patient (turn {self.turn_count}){meta_tag}:{C.RESET} {text}")

    def _print_summary(self, summary: dict):
        print(f"\n{C.BOLD}{'─'*60}")
        print(f"  SUMMARY")
        print(f"{'─'*60}{C.RESET}")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    def _error(self, msg: str):
        self.errors.append(msg)
        print(f"  {C.RED}✗ {msg}{C.RESET}")

    def _finalize(self) -> Tuple[bool, Dict]:
        # Compute stats
        questions_asked = {q: a for q, a in self._question_attempts.items()}
        total_questions = len(questions_asked)
        first_attempt = sum(1 for a in questions_asked.values() if a == 1)
        re_asked = {q: a for q, a in questions_asked.items() if a > 1}
        first_attempt_rate = (first_attempt / total_questions * 100) if total_questions > 0 else 0

        elapsed = time.time() - self._start_time

        stats = {
            "persona": self.persona_key,
            "run": self.run_number,
            "total_turns": self.turn_count,
            "questions_asked": total_questions,
            "first_attempt_accepted": first_attempt,
            "first_attempt_rate": round(first_attempt_rate, 1),
            "re_asked": dict(re_asked),
            "free_chat_turns": self.free_chat_turns,
            "fallbacks": self.fallback_count,
            "errors": list(self.errors),
            "passed": len(self.errors) == 0,
            "turns_per_question": dict(sorted(questions_asked.items())),
            "elapsed_s": round(elapsed, 1),
        }

        # Format duration
        minutes, secs = divmod(int(elapsed), 60)
        time_str = f"{minutes}m {secs}s" if minutes else f"{secs}s"

        print(f"\n{C.BOLD}{'='*60}")
        print(f"  RESULTS — {self.persona['name']} Run {self.run_number}")
        print(f"{'='*60}{C.RESET}")
        print(f"  Total turns:              {self.turn_count}")
        print(f"  Questions asked:          {total_questions}")
        print(f"  First-attempt accepted:   {first_attempt}/{total_questions} ({first_attempt_rate:.0f}%)")
        print(f"  Re-asked questions:       {len(re_asked)}")
        print(f"  Free chat turns:          {self.free_chat_turns}")
        print(f"  Fallbacks:                {self.fallback_count}")
        print(f"  Errors:                   {len(self.errors)}")
        print(f"  Duration:                 {time_str}")

        if re_asked:
            print(f"\n  {C.YELLOW}Re-asked questions:{C.RESET}")
            for qid, attempts in re_asked.items():
                print(f"    {qid}: {attempts} attempts")

        if self.errors:
            print()
            for e in self.errors:
                print(f"  {C.RED}✗ {e}{C.RESET}")
            print(f"\n  {C.RED}{C.BOLD}FAILED{C.RESET}\n")
            return False, stats
        else:
            print(f"\n  {C.GREEN}{C.BOLD}PASSED{C.RESET}\n")
            return True, stats


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AnæstesiCare evaluation tester — runs personas N times with API-driven question routing"
    )
    parser.add_argument(
        "--persona", "-p",
        choices=list(PERSONAS.keys()) + ["all"],
        default="uncertain",
        help="Patient persona to simulate (default: uncertain)",
    )
    parser.add_argument(
        "--runs", "-n",
        type=int,
        default=1,
        help="Number of times to run each persona (default: 1)",
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
            q_count = len(persona["responses"])
            print(f"  {key:15s} — {persona['name']} ({q_count} question responses)")
        print(f"\nTranscripts saved to: {TRANSCRIPT_DIR}")
        print()
        sys.exit(0)

    personas_to_run = list(PERSONAS.keys()) if args.persona == "all" else [args.persona]

    # Collect all stats: { persona_key: [stats_dict, stats_dict, ...] }
    all_stats: Dict[str, List[Dict]] = {p: [] for p in personas_to_run}

    def print_persona_summary(persona_key: str, runs: List[Dict], out_fn=print):
        """Print summary table for one persona's runs."""
        passed = sum(1 for r in runs if r["passed"])
        total = len(runs)

        turns_list = [r["total_turns"] for r in runs]
        fa_rates = [r["first_attempt_rate"] for r in runs]
        avg_turns = sum(turns_list) / len(turns_list) if turns_list else 0
        avg_fa = sum(fa_rates) / len(fa_rates) if fa_rates else 0
        min_turns = min(turns_list) if turns_list else 0
        max_turns_val = max(turns_list) if turns_list else 0

        elapsed_list = [r.get("elapsed_s", 0) for r in runs]
        total_elapsed = sum(elapsed_list)
        avg_elapsed = total_elapsed / len(elapsed_list) if elapsed_list else 0
        total_m, total_s = divmod(int(total_elapsed), 60)
        avg_m, avg_s = divmod(int(avg_elapsed), 60)
        total_time_str = f"{total_m}m {total_s}s" if total_m else f"{total_s}s"
        avg_time_str = f"{avg_m}m {avg_s}s" if avg_m else f"{avg_s}s"

        out_fn(f"  {'─'*56}")
        out_fn(f"  {C.BOLD}{persona_key.upper()}{C.RESET} ({PERSONAS[persona_key]['name']})")
        out_fn(f"  {'─'*56}")
        out_fn(f"  Completed:              {passed}/{total}")
        out_fn(f"  Avg turns:              {avg_turns:.1f}  (min {min_turns}, max {max_turns_val})")
        out_fn(f"  Avg first-attempt rate: {avg_fa:.1f}%")
        out_fn(f"  Avg duration:           {avg_time_str}  (total {total_time_str})")

        # Per-run table
        out_fn()
        out_fn(f"  {'Run':<6} {'Turns':<8} {'1st-attempt':<20} {'Re-asked':<30} {'Time':<8} {'Result'}")
        out_fn(f"  {'───':<6} {'─────':<8} {'───────────':<20} {'────────':<30} {'────':<8} {'──────'}")

        for r in runs:
            re_asked_str = ", ".join(f"{q}({a}x)" for q, a in r["re_asked"].items()) if r["re_asked"] else "—"
            status = f"{C.GREEN}PASS{C.RESET}" if r["passed"] else f"{C.RED}FAIL{C.RESET}"
            fa_str = f"{r['first_attempt_accepted']}/{r['questions_asked']} ({r['first_attempt_rate']}%)"
            elapsed = r.get("elapsed_s", 0)
            m, s = divmod(int(elapsed), 60)
            time_str = f"{m}m{s:02d}s" if m else f"{s}s"
            out_fn(f"  {r['run']:<6} {r['total_turns']:<8} {fa_str:<20} {re_asked_str:<30} {time_str:<8} {status}")

        # Which questions get re-asked most often
        reask_totals: Dict[str, int] = {}
        for r in runs:
            for qid, attempts in r["re_asked"].items():
                reask_totals[qid] = reask_totals.get(qid, 0) + 1

        if reask_totals:
            out_fn()
            out_fn(f"  Questions re-asked (across {total} runs):")
            for qid, count in sorted(reask_totals.items(), key=lambda x: -x[1]):
                out_fn(f"    {qid}: re-asked in {count}/{total} runs ({count/total*100:.0f}%)")

        out_fn()

    # ==================================================================
    # Run all personas
    # ==================================================================

    for persona_key in personas_to_run:
        for run_num in range(1, args.runs + 1):
            print(f"\n{'#' * 60}")
            print(f"  {persona_key.upper()} — Run {run_num}/{args.runs}")
            print(f"{'#' * 60}")

            tester = ConversationTester(
                base_url=args.base_url,
                persona_key=persona_key,
                run_number=run_num,
                delay=args.delay,
            )
            passed, stats = tester.run()
            all_stats[persona_key].append(stats)

            if run_num < args.runs:
                time.sleep(1)

        # ── Per-persona summary (printed immediately after all runs for this persona) ──
        print(f"\n{'='*60}")
        print(f"  PERSONA SUMMARY — {persona_key.upper()}")
        print(f"{'='*60}")
        print_persona_summary(persona_key, all_stats[persona_key])

        # Save per-persona summary to file
        os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
        persona_summary_lines = []
        def _capture(line: str = ""):
            clean = line
            for code in [C.BLUE, C.GREEN, C.YELLOW, C.RED, C.GRAY, C.BOLD, C.RESET]:
                clean = clean.replace(code, "")
            persona_summary_lines.append(clean)

        _capture(f"PERSONA SUMMARY — {persona_key.upper()}")
        _capture(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        _capture(f"Runs: {args.runs}")
        _capture("")
        print_persona_summary(persona_key, all_stats[persona_key], out_fn=_capture)

        persona_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        persona_summary_path = os.path.join(
            TRANSCRIPT_DIR, f"__{persona_key}__summary__{persona_ts}.txt"
        )
        with open(persona_summary_path, "w", encoding="utf-8") as pf:
            pf.write("\n".join(persona_summary_lines))
        print(f"  {C.GRAY}📄 Persona summary saved: {persona_summary_path}{C.RESET}")

    # ==================================================================
    # Final summary — all personas combined, printed and saved to file
    # ==================================================================

    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(TRANSCRIPT_DIR, f"__summary__{timestamp}.txt")

    summary_lines = []

    def out(line: str = ""):
        """Print to terminal and append to summary file."""
        print(line)
        clean = line
        for code in [C.BLUE, C.GREEN, C.YELLOW, C.RED, C.GRAY, C.BOLD, C.RESET]:
            clean = clean.replace(code, "")
        summary_lines.append(clean)

    out(f"\n{'='*60}")
    out(f"  FULL EVALUATION SUMMARY")
    out(f"{'='*60}")
    out(f"  Date:             {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    out(f"  Runs per persona: {args.runs}")
    out(f"  Transcripts dir:  {TRANSCRIPT_DIR}")
    out()

    # ── Per-persona detail (repeated in saved file) ──
    for persona_key in personas_to_run:
        print_persona_summary(persona_key, all_stats[persona_key], out_fn=out)

    # ── Cross-persona comparison table ──
    out(f"  {'='*56}")
    out(f"  CROSS-PERSONA COMPARISON")
    out(f"  {'='*56}")
    out()
    out(f"  {'Persona':<18} {'Completed':<12} {'Avg turns':<12} {'Avg 1st-att':<14} {'Avg time':<10} {'Most re-asked'}")
    out(f"  {'───────':<18} {'─────────':<12} {'─────────':<12} {'───────────':<14} {'────────':<10} {'─────────────'}")

    total_passed = 0
    total_runs = 0
    grand_total_elapsed = 0

    for persona_key in personas_to_run:
        runs = all_stats[persona_key]
        passed = sum(1 for r in runs if r["passed"])
        total = len(runs)
        total_passed += passed
        total_runs += total

        turns_list = [r["total_turns"] for r in runs]
        fa_rates = [r["first_attempt_rate"] for r in runs]
        elapsed_list = [r.get("elapsed_s", 0) for r in runs]
        avg_turns = sum(turns_list) / len(turns_list) if turns_list else 0
        avg_fa = sum(fa_rates) / len(fa_rates) if fa_rates else 0
        avg_elapsed = sum(elapsed_list) / len(elapsed_list) if elapsed_list else 0
        grand_total_elapsed += sum(elapsed_list)

        m, s = divmod(int(avg_elapsed), 60)
        avg_time_str = f"{m}m{s:02d}s" if m else f"{s}s"

        # Find the most frequently re-asked question across all runs
        reask_totals: Dict[str, int] = {}
        for r in runs:
            for qid in r["re_asked"]:
                reask_totals[qid] = reask_totals.get(qid, 0) + 1
        if reask_totals:
            worst_q = max(reask_totals, key=reask_totals.get)
            worst_str = f"{worst_q} ({reask_totals[worst_q]}/{total})"
        else:
            worst_str = "—"

        out(f"  {persona_key:<18} {passed}/{total:<10} {avg_turns:<12.1f} {avg_fa:<14.1f}% {avg_time_str:<10} {worst_str}")

    out()
    out(f"  {'─'*56}")
    overall_rate = (total_passed / total_runs * 100) if total_runs > 0 else 0
    grand_m, grand_s = divmod(int(grand_total_elapsed), 60)
    grand_time_str = f"{grand_m}m {grand_s}s" if grand_m else f"{grand_s}s"
    out(f"  OVERALL: {total_passed}/{total_runs} completed ({overall_rate:.0f}%)  —  Total time: {grand_time_str}")
    out(f"  {'─'*56}")
    out()

    # Write summary file
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"  {C.GRAY}📄 Summary saved: {summary_path}{C.RESET}\n")

    sys.exit(0 if total_passed == total_runs else 1)


if __name__ == "__main__":
    main()