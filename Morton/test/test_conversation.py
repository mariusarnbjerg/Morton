"""
Automated conversation tester used for Evaluation.

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

PERSONAS_1: Dict[str, Dict] = {

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
            "q2.1": ["About 173 I think"],
            "q2.2": ["Maybe 80 kilos?"],
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

"""
Evaluation personas for 2nd iteration testing — REVISED.

These personas are derived from behavioral patterns observed across 20
real user transcripts from the 1st iteration test. Each response is
sourced from or inspired by a specific transcript, referenced in comments.

IMPORTANT: These personas use DIFFERENT phrasings from those that informed
the completion criteria refinements, to avoid overfitting the evaluation
to the specific language the criteria were tuned for.

Source transcript IDs use the last 3 digits for brevity:
  #382 = 1772382629904, #379 = 1772379728253, etc.
"""

PERSONAS_2: Dict[str, Dict] = {

    # ==================================================================
    # PERSONA 1: UNCERTAIN / VAGUE
    #
    # Tests: completion criteria with hedging language, "I don't know"
    # responses, forgotten details, and directional-but-unclear answers.
    #
    # All phrasings differ from those used during criteria refinement.
    # ==================================================================
    "uncertain": {
        "name": "Uncertain Patient",
        "responses": {
            "q0": ["sure I guess"],  # inspired by #188 "Sure"
            "q1": ["Henrik"],
            "q2": ["55 I believe"],  # inspired by #185 "oh sorry, my age is actually 30" (uncertain about own details)
            "q2.1": ["About 173 I think"],
            "q2.2": ["Maybe 80 kilos?"],
            "q3": ["hmm, I'm not sure.. maybe?"],
            # inspired by #185 "I'm not sure.." and #188 "I dont think so, but Im unsure"
            "q3b": ["I honestly cant remember, it was years ago"],  # inspired by #179 "Not in particular"
            "q4": ["I assume not, I am not sure"],
            # from #179 verbatim (different from q3 phrasing used in criteria fix)
            "q4b": ["I really dont know the details"],  # inspired by #374 "the unknown situation"
            "q5": ["I dont think I do? this is kind of new to me"],
            # inspired by #185 "I don't think I do? I'm not sure. this is my first time."
            "q6": ["I havent been diagnosed but people tell me I stop breathing at night"],
            # from #382 (different phrasing than "I snore alot" used in old persona)
            "q7": ["no I dont believe so"],  # inspired by #191 "not that i know of"
            "q8": ["none that I'm aware of"],  # inspired by #202 "Not that I know of"
            "q8b": ["I honestly cant say for certain"],  # inspired by #179 "I assume not"
            "q9": ["I do take something but I cant remember what its called"],
            # from #198 "yes but I don't remember the name" (different phrasing than old persona)
            "q9b": ["its for something with my stomach I think, I really dont know the name"],
            # inspired by #377a "I dontt know what kind of medication it is"
            "q10": ["not as far as I know"],  # inspired by #191 "Not as far as i know"
            "q11": ["I take some vitamins when I remember to"],  # from #191 "I take multivitamins the days i remember"
            "q11b": ["just regular vitamins I think, nothing special"],  # inspired by #374 "only normal vitamins"
            "q12": ["not that I'm aware of no"],  # inspired by #371 "not to my knowledge"
            "q13": ["not really I dont think so"],  # inspired by #188 "Not really"
            "q14": ["sometimes a little, but I think thats just because im out of shape"],
            # from #192 (different phrasing than old persona)
            "q15": ["I get heartburn occasionally but its nothing serious I think"],
            # inspired by #358 "I do have a heartburn sometimes but nothing serious"
            "q16": ["no I dont believe so"],  # inspired by #179 "not in particular"
            "q17": ["no not anymore"],  # from #219 "no not anymore"
            "q17b": ["I stopped a while back, cant remember exactly when"],  # inspired by #219 quitting context
            "q18": ["not really that much"],  # from #291 "Not really that much"
            "q19": ["no I dont think so"],  # inspired by #201 "no i dont think so"
            "q19b": ["no"],
            "q20": ["not really, I think it will be fine"],  # inspired by #192 "not really, i trust the doctors"
            "q21": ["I cant think of anything right now"],  # inspired by #179 "I do not think so"
            "q22": ["no I believe we covered it all"],  # inspired by #191 "no i dont think so"
        },
    },

    # ==================================================================
    # PERSONA 2: INQUISITIVE / COMPOUND
    #
    # Tests: answer extraction from compound messages (answer + question),
    # medical scope boundary, off-topic resilience.
    #
    # All phrasings differ from the old persona to avoid overlap with
    # criteria examples.
    # ==================================================================
    "inquisitive": {
        "name": "Inquisitive Patient",
        "responses": {
            "q0": ["yes I'm ready, how long will this take by the way?"],
            # inspired by #379 "Yes - what do you want to know about?"
            "q1": ["Maria, what is this information used for exactly?"],
            # inspired by #377b "can you tell me about the surgical procedure"
            "q2": ["44, does age matter for the anesthesia?"],  # inspired by #377b "Why do you need my age?"
            "q3": ["I had it once for a tooth extraction, what kind will I be getting?"],
            # inspired by #218 "yes, but only when I gave birth... only from my stomach and down"
            "q3b": ["no it went fine, but should I expect side effects this time?"],
            # inspired by #377b "No, but should i be worried?"
            "q4": ["not that I know of, is that something that runs in families?"],
            # inspired by #205 "my mother once got a rash after, should i be concerned?"
            "q4b": ["I said I dont know of any"],
            "q5": ["no, what would happen if I did have a difficult airway?"],
            # inspired by #377b "Will I be intubated?"
            "q6": ["no I dont, but what is a CPAP machine exactly?"],
            # inspired by #382 "What does it mean if I have sleep apnea?"
            "q7": ["I have some metal wires behind my teeth from braces, is that a concern?"],
            # from #202 "I have a metal wire on the back of both my rows of teeth"
            "q8": ["yes I react to some types of tape, could that be a problem during surgery?"],
            # inspired by #377a latex allergy context
            "q8b": ["medical tape gives me a rash, do you use that during the operation?"],
            # inspired by #377a "whatever allergy that XXS condoms contain"
            "q9": ["yes I take eye drops that are prescription, do those count?"],
            # from #202 "During summer I take eye drops which are prescription"
            "q9b": ["prescription eye drops for allergies, I dont know the brand name"],
            # inspired by #198 "yes but I don't remember the name"
            "q10": ["no, what are those used for exactly?"],  # from #377b "what is warfarin used for?"
            "q11": ["I take creatine and some hair vitamins, should I stop those before surgery?"],
            # from #202 "I take creatine, 6g daily and vitamins for nails and hair"
            "q11b": ["creatine, hair vitamins, and sometimes D vitamin"],  # from #202
            "q12": ["no, but can anesthesia affect the heart?"],  # inspired by #377b concern-style questions
            "q13": ["no but what is COPD? I keep hearing that word"],
            # inspired by #371 "what is copd?" and #295 "I do not know what anesthesia is"
            "q14": ["only if I really push myself like running, is that a problem?"],
            # inspired by #185 "yes, after 20km"
            "q15": ["I get it sometimes when I drink wine, is that relevant?"],
            # from #218 "2 times a month... when drinking wine"
            "q16": ["I had a cold a while back but I feel fine now, does that matter?"],
            # inspired by #358 "I only had a cold 2 weeks ago"
            "q17": ["no never, but my partner smokes inside, is passive smoking a concern?"],
            # inspired by general concern pattern
            "q17b": ["I said I dont smoke"],
            "q18": ["maybe 2 glasses on the weekend, is that too much before surgery?"],
            # inspired by #371 "in what unit?"
            "q19": ["no, but can recreational drugs interact with anesthesia?"],
            # inspired by #205 "can i be charged by what i say here"
            "q19b": ["no I said I dont use any"],
            "q20": ["a bit yes, what happens if I wake up during the procedure?"],
            # inspired by #185 "I am. I will be unconscious. I'm afraid i'll be ded"
            "q21": ["I donate blood regularly, should I mention that?"],  # from #202 "I donate blood"
            "q22": ["yes, how soon before surgery do I need to stop eating?"],
            # inspired by #382 fasting question context
        },
    },

    # ==================================================================
    # PERSONA 3: CONTRADICTORY / CORRECTING
    #
    # Tests: state management with self-corrections, accidental yes/no,
    # retroactive information, frustration with re-asking.
    #
    # Phrasings are drawn from different transcripts than the old persona.
    # ==================================================================
    "contradictory": {
        "name": "Contradictory Patient",
        "responses": {
            "q0": ["lets go"],
            "q1": ["Katrine Holm"],
            "q2": ["oh sorry I'm 31, I keep saying 30 out of habit"],  # from #185 "oh sorry, my age is actually 30"
            "q3": ["yes, well actually I think it was local anesthesia not full"],
            # from #218 "yes, but only when I gave birth... only from my stomach and down"
            "q3b": [
                "no wait, I'm not sure it counts as real anesthesia actually",
                # inspired by #192 "wait i forgot i had been under anesthesia i think"
                "I think it was just local, no problems with it though",
            ],
            "q4": ["hmm yes I think my aunt had something"],  # inspired by #205 "my mother once got a rash after"
            "q4b": [
                "actually I might be mixing it up with something else, I'm not confident",
                # inspired by #374 "the unknown situation"
                "honestly I dont remember, it might not have been anesthesia related",
            ],
            "q5": ["no, well I was told my throat is narrow on one side"],
            # from #218 "yes in one side of my noise" and #202 "enlarged epiglottis"
            "q6": ["no, actually my husband says I stop breathing sometimes at night"],
            # inspired by #291 "Might have sleep apnea"
            "q7": ["no, oh wait I do have crowns on some teeth"],  # inspired by #218 "I have kroner på tænderne"
            "q8": ["no, well actually I react to some plasters but I wouldnt call it an allergy"],
            # inspired by #205 "well, not that i know of"
            "q8b": [
                "I said I dont think its a real allergy, my skin just gets irritated",
                "just irritation from adhesive plasters, nothing serious",
            ],
            "q9": ["no, oh wait I do take something for my thyroid I forgot about that"],
            # inspired by #192 "wait i forgot" pattern
            "q9b": ["thyroid medication, I always forget to count it"],
            "q10": ["no I dont, actually wait does ibuprofen count as a blood thinner?"],
            # inspired by #371 "What if I take medications every second day?"
            "q11": ["no, well I take painkillers when I get headaches"],
            # from #377b "No, but sometimes i get headaches and take medications"
            "q11b": ["just panodil for headaches, nothing else"],  # inspired by #382 "panodiler"
            "q12": [
                "yes",
                "no sorry I was thinking of my father, he has heart issues not me",
                # inspired by #382 "No, I don't have any problems with my heart health - you got that wrong"
            ],
            "q13": ["I have asthma, well had asthma, I havent used my inhaler in years"],
            # inspired by #219 "no not anymore" pattern
            "q14": ["no, well maybe a little when climbing many stairs"],
            # inspired by #192 "Sometimes a little, but I think thats just because Im not in the best shape"
            "q15": ["no, actually yes sometimes but only after eating late"],
            # inspired by #218 "yes sometimes acid reflux"
            "q16": ["no, oh actually I had a bad cold two weeks ago but I'm over it now"],
            # from #358 "I only had a cold 2 weeks ago"
            "q17": ["I used to smoke but I stopped, well I still have one at parties sometimes"],
            # inspired by #192 "only at parties"
            "q17b": ["smoked for maybe 5 years, quit 3 years ago, now just socially"],
            "q18": ["I dont drink, well maybe a beer or two on fridays"],
            # inspired by #185 "Why do you assume i drank alcohol? I did not" then later admitting some
            "q19": ["yes"],
            "q19b": ["actually no, I dont use anything"],
            "q20": ["no im fine, well actually I am a little worried about waking up during it"],
            # inspired by #185 "I am. I will be unconscious. I'm afraid i'll be ded"
            "q21": ["no wait I forgot, I had my tonsils removed as a kid, does that matter?"],
            # from #192 "wait i forgot i had been under anesthesia i think when i got rid of my tonsils"
            "q22": ["no that should be everything, well actually who reads all this afterwards?"],
            # inspired by #205 "so this will not be shared to the authorities?"
        },
    },
}

PERSONAS: Dict[str, Dict] = {

    # ------------------------------------------------------------------
    # Patient 01 | DurableKey 5693 | Age 19 | Sex Female | ASA 1
    # ------------------------------------------------------------------
    "patient_01": {
        "name": "Emma",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Emma"],
            "q2": ["I am 19 years old"],
            "q2.1": ["160 cm"],
            "q2.2": ["49.5 kg"],
            "q2.3": ["Female"],

            "q3": ["No"],
            "q3b": ["I have not had anesthesia before"],
            "q4": ["No"],
            "q4b": ["No known family reactions"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["Yes"],
            "q8b": ["I react to NSAID anti-inflammatory painkillers, but I am not sure about the exact reaction"],
            "q9": ["No"],
            "q9b": ["No daily prescription medication"],
            "q10": ["No"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["No"],
            "q13": ["No"],
            "q14": ["No"],
            "q15": ["No"],
            "q16": ["No"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine either"],
            "q18": ["I do not really drink alcohol"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["Yes, I can feel a bit anxious"],
            "q21": ["I have had anxiety, and I have had a breast lump checked"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 02 | DurableKey 5573 | Age 39 | Sex Female | ASA 1
    # ------------------------------------------------------------------
    "patient_02": {
        "name": "Sofia",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Sofia"],
            "q2": ["I am 39 years old"],
            "q2.1": ["169 cm"],
            "q2.2": ["80 kg"],
            "q2.3": ["Female"],

            "q3": ["No"],
            "q3b": ["I have not had anesthesia before"],
            "q4": ["No"],
            "q4b": ["No known family reactions"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["No"],
            "q8b": ["No known allergies"],
            "q9": ["No"],
            "q9b": ["No daily prescription medication"],
            "q10": ["No"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["No"],
            "q13": ["Yes, I have asthma"],
            "q14": ["No"],
            "q15": ["No"],
            "q16": ["No"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I drink occasionally, but not much"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["A little, mostly because I have not tried anesthesia before"],
            "q21": ["I have asthma, and I am being treated because of a missed abortion"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 03 | DurableKey 40179 | Age 49 | Sex Female | ASA 1
    # ------------------------------------------------------------------
    "patient_03": {
        "name": "Laura",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Laura"],
            "q2": ["I am 49 years old"],
            "q2.1": ["167 cm"],
            "q2.2": ["61 kg"],
            "q2.3": ["Female"],

            "q3": ["No"],
            "q3b": ["I have not had anesthesia before"],
            "q4": ["Uncertain"],
            "q4b": ["I have never heard of anyone in my family reacting badly to anesthesia"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["No"],
            "q8b": ["No known allergies"],
            "q9": ["No"],
            "q9b": ["No daily prescription medication"],
            "q10": ["No"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["No"],
            "q13": ["No"],
            "q14": ["No"],
            "q15": ["No"],
            "q16": ["No"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I drink occasionally"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["No, I feel okay about it"],
            "q21": ["I have osteoarthritis"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 04 | DurableKey 2269 | Age 61 | Sex Female | ASA 1
    # ------------------------------------------------------------------
    "patient_04": {
        "name": "Hanne",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Hanne"],
            "q2": ["I am 61 years old"],
            "q2.1": ["168 cm"],
            "q2.2": ["80 kg"],
            "q2.3": ["Female"],

            "q3": ["Yes"],
            "q3b": ["No problems that I know of"],
            "q4": ["No"],
            "q4b": ["No known family reactions"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["Yes"],
            "q8b": ["I react to tramadol and sulfamethizole"],
            "q9": ["No"],
            "q9b": ["No daily prescription medication"],
            "q10": ["No"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["No"],
            "q13": ["No"],
            "q14": ["No"],
            "q15": ["No"],
            "q16": ["No"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I do not really drink alcohol"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["No, I feel okay about it"],
            "q21": ["I have had prolapse problems, a cystocele, and stress incontinence"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 05 | DurableKey 28435 | Age 83 | Sex Female | ASA 1
    # ------------------------------------------------------------------
    "patient_05": {
        "name": "Ingrid",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Ingrid"],
            "q2": ["I am 83 years old"],
            "q2.1": ["166 cm"],
            "q2.2": ["52 kg"],
            "q2.3": ["Female"],

            "q3": ["Yes"],
            "q3b": ["No problems that I know of"],
            "q4": ["Uncertain"],
            "q4b": ["I do not know of any family reactions"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["No"],
            "q8b": ["No known allergies"],
            "q9": ["No"],
            "q9b": ["No daily prescription medication"],
            "q10": ["No"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["No"],
            "q13": ["No"],
            "q14": ["No"],
            "q15": ["No"],
            "q16": ["Yes, I have an infection around a joint prosthesis"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I do not really drink alcohol"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["No, I am not especially worried"],
            "q21": ["I have an infection around a joint prosthesis after a previous operation"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 06 | DurableKey 10293 | Age 24 | Sex Male | ASA 1
    # ------------------------------------------------------------------
    "patient_06": {
        "name": "Magnus",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Magnus"],
            "q2": ["I am 24 years old"],
            "q2.1": ["192 cm"],
            "q2.2": ["100 kg"],
            "q2.3": ["Male"],

            "q3": ["Yes"],
            "q3b": ["No problems, but I have been told I should not have a spinal block because of my back"],
            "q4": ["No"],
            "q4b": ["No known family reactions"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["Yes"],
            "q8b": ["I react to morphine"],
            "q9": ["No"],
            "q9b": ["No daily prescription medication"],
            "q10": ["No"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["No"],
            "q13": ["No"],
            "q14": ["No"],
            "q15": ["No"],
            "q16": ["No"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I drink occasionally, mostly socially"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["No, I feel okay about it"],
            "q21": ["I have a fracture in my ankle area, and I have a shortened spinal cord according to my mother"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 07 | DurableKey 14653 | Age 43 | Sex Male | ASA 1
    # ------------------------------------------------------------------
    "patient_07": {
        "name": "Jonas",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Jonas"],
            "q2": ["I am 43 years old"],
            "q2.1": ["179 cm"],
            "q2.2": ["84 kg"],
            "q2.3": ["Male"],

            "q3": ["Yes"],
            "q3b": ["No problems"],
            "q4": ["No"],
            "q4b": ["No known family reactions"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["No"],
            "q8b": ["No known allergies"],
            "q9": ["No"],
            "q9b": ["No daily prescription medication"],
            "q10": ["No"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["No"],
            "q13": ["No"],
            "q14": ["No"],
            "q15": ["No"],
            "q16": ["No"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I drink occasionally"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["No, I feel fine about it"],
            "q21": ["I have a previous injury in my lower leg"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 08 | DurableKey 6997 | Age 56 | Sex Male | ASA 1
    # ------------------------------------------------------------------
    "patient_08": {
        "name": "Lars",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Lars"],
            "q2": ["I am 56 years old"],
            "q2.1": ["183 cm"],
            "q2.2": ["90 kg"],
            "q2.3": ["Male"],

            "q3": ["Yes"],
            "q3b": ["No problems"],
            "q4": ["Uncertain"],
            "q4b": ["I have never heard of any family reactions to anesthesia"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["No"],
            "q8b": ["No known allergies"],
            "q9": ["No"],
            "q9b": ["No daily prescription medication"],
            "q10": ["No"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["No"],
            "q13": ["No"],
            "q14": ["No"],
            "q15": ["No"],
            "q16": ["No"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I drink occasionally, but not a lot"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["No, I am not worried"],
            "q21": ["I have polyps or disease in the gallbladder"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 09 | DurableKey 13048 | Age 74 | Sex Male | ASA 1
    # ------------------------------------------------------------------
    "patient_09": {
        "name": "Erik",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Erik"],
            "q2": ["I am 74 years old"],
            "q2.1": ["175 cm"],
            "q2.2": ["82 kg"],
            "q2.3": ["Male"],

            "q3": ["No"],
            "q3b": ["I have not had anesthesia before"],
            "q4": ["No"],
            "q4b": ["No known family reactions"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["No"],
            "q8b": ["No known allergies"],
            "q9": ["No"],
            "q9b": ["No daily prescription medication"],
            "q10": ["No"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["No"],
            "q13": ["No"],
            "q14": ["No"],
            "q15": ["No"],
            "q16": ["No"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I do not really drink alcohol"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["A little, mostly because I have not tried anesthesia before"],
            "q21": ["I have osteoarthritis in both knees"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 10 | DurableKey 4162 | Age 26 | Sex Female | ASA 2
    # ------------------------------------------------------------------
    "patient_10": {
        "name": "Freja",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Freja"],
            "q2": ["I am 26 years old"],
            "q2.1": ["158 cm"],
            "q2.2": ["76 kg"],
            "q2.3": ["Female"],

            "q3": ["No"],
            "q3b": ["I have not had anesthesia before"],
            "q4": ["No"],
            "q4b": ["No known family reactions"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["Yes"],
            "q8b": ["I react to oxycodone"],
            "q9": ["No"],
            "q9b": ["No daily prescription medication"],
            "q10": ["No"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["No"],
            "q13": ["No"],
            "q14": ["No"],
            "q15": ["No"],
            "q16": ["No"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I drink occasionally"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["A little, mostly because I have not tried anesthesia before"],
            "q21": ["I have a shoulder problem, including a SLAP lesion"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 11 | DurableKey 13812 | Age 44 | Sex Female | ASA 2
    # ------------------------------------------------------------------
    "patient_11": {
        "name": "Maria",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Maria"],
            "q2": ["I am 44 years old"],
            "q2.1": ["153 cm"],
            "q2.2": ["61 kg"],
            "q2.3": ["Female"],

            "q3": ["Yes"],
            "q3b": ["No problems"],
            "q4": ["No"],
            "q4b": ["No known family reactions"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["No"],
            "q8b": ["No known allergies"],
            "q9": ["No"],
            "q9b": ["No daily prescription medication"],
            "q10": ["No"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["No"],
            "q13": ["No"],
            "q14": ["No"],
            "q15": ["No"],
            "q16": ["No"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I drink occasionally, mostly socially"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["No, I feel okay about it"],
            "q21": ["I have heavy or irregular menstrual bleeding"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 12 | DurableKey 4460 | Age 62 | Sex Female | ASA 2
    # ------------------------------------------------------------------
    "patient_12": {
        "name": "Lone",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Lone"],
            "q2": ["I am 62 years old"],
            "q2.1": ["I am not sure"],
            "q2.2": ["I am not sure"],
            "q2.3": ["Female"],

            "q3": ["Yes"],
            "q3b": ["No problems that I know of"],
            "q4": ["Uncertain"],
            "q4b": ["I do not know of any family reactions"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["No"],
            "q8b": ["No known allergies"],
            "q9": ["Yes"],
            "q9b": ["I take antidepressant medication, and I am treated for low thyroid function"],
            "q10": ["No"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["Yes, I have high blood pressure"],
            "q13": ["No"],
            "q14": ["No"],
            "q15": ["No"],
            "q16": ["No"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I do not really drink alcohol"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["A little, mostly because of the cancer diagnosis"],
            "q21": ["I have breast cancer, periods with depression, high blood pressure, and low thyroid function"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 13 | DurableKey 27355 | Age 93 | Sex Female | ASA 2
    # ------------------------------------------------------------------
    "patient_13": {
        "name": "Gerda",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Gerda"],
            "q2": ["I am 93 years old"],
            "q2.1": ["I am not sure"],
            "q2.2": ["I am not sure"],
            "q2.3": ["Female"],

            "q3": ["Yes"],
            "q3b": ["No problems that I know of"],
            "q4": ["Uncertain"],
            "q4b": ["I do not know of any family reactions"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["No"],
            "q8b": ["No known allergies"],
            "q9": ["Yes"],
            "q9b": ["I take Pradaxa"],
            "q10": ["Yes, I take Pradaxa"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["Yes, I have atrial fibrillation, and possibly high blood pressure"],
            "q13": ["No"],
            "q14": ["No"],
            "q15": ["Yes, I have reflux"],
            "q16": ["No"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I do not really drink alcohol"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["No, I am not especially worried"],
            "q21": ["I am being examined because of suspected cancer, and I have atrial fibrillation and reflux"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 14 | DurableKey 11614 | Age 24 | Sex Male | ASA 2
    # ------------------------------------------------------------------
    "patient_14": {
        "name": "Oliver",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Oliver"],
            "q2": ["I am 24 years old"],
            "q2.1": ["186 cm"],
            "q2.2": ["106.6 kg"],
            "q2.3": ["Male"],

            "q3": ["No"],
            "q3b": ["I have not had anesthesia before"],
            "q4": ["No"],
            "q4b": ["No known family reactions"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["No"],
            "q8b": ["No known allergies"],
            "q9": ["No"],
            "q9b": ["No daily prescription medication"],
            "q10": ["No"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["No"],
            "q13": ["No"],
            "q14": ["No"],
            "q15": ["No"],
            "q16": ["No"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I drink occasionally, mostly socially"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["A little, mostly because I have not tried anesthesia before"],
            "q21": ["I have an inguinal hernia and have also had a bruised toe"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 15 | DurableKey 6417 | Age 39 | Sex Male | ASA 2
    # ------------------------------------------------------------------
    "patient_15": {
        "name": "Mads",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Mads"],
            "q2": ["I am 39 years old"],
            "q2.1": ["180 cm"],
            "q2.2": ["63 kg"],
            "q2.3": ["Male"],

            "q3": ["Yes"],
            "q3b": ["No problems that I know of"],
            "q4": ["No"],
            "q4b": ["No known family reactions"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["No"],
            "q8b": ["No known allergies"],
            "q9": ["Yes"],
            "q9b": ["I have epilepsy, but I am not sure what medication is relevant here"],
            "q10": ["No"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["No"],
            "q13": ["No"],
            "q14": ["No"],
            "q15": ["No"],
            "q16": ["No"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I drink occasionally"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["A little, mostly because of the testicular cancer examination"],
            "q21": ["I have epilepsy and I am being examined because of suspected testicular cancer"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 16 | DurableKey 7884 | Age 57 | Sex Male | ASA 2
    # ------------------------------------------------------------------
    "patient_16": {
        "name": "Thomas",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Thomas"],
            "q2": ["I am 57 years old"],
            "q2.1": ["165 cm"],
            "q2.2": ["95 kg"],
            "q2.3": ["Male"],

            "q3": ["No"],
            "q3b": ["I have not had anesthesia before"],
            "q4": ["Uncertain"],
            "q4b": ["I have never heard of any family reactions"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["No"],
            "q8b": ["No known allergies"],
            "q9": ["No"],
            "q9b": ["No daily prescription medication"],
            "q10": ["No"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["No"],
            "q13": ["No"],
            "q14": ["No, I can walk up to the second floor without a break"],
            "q15": ["No"],
            "q16": ["No"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I drink occasionally, but not much"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["No, I feel okay about it"],
            "q21": ["I have gallstones"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 17 | DurableKey 6574 | Age 61 | Sex Male | ASA 2
    # ------------------------------------------------------------------
    "patient_17": {
        "name": "Henrik",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Henrik"],
            "q2": ["I am 61 years old"],
            "q2.1": ["183 cm"],
            "q2.2": ["94 kg"],
            "q2.3": ["Male"],

            "q3": ["Yes"],
            "q3b": ["No problems that I know of"],
            "q4": ["Uncertain"],
            "q4b": ["I do not know of any family reactions"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["No"],
            "q8b": ["No known allergies"],
            "q9": ["Yes"],
            "q9b": ["I am on blood-thinning treatment, but I do not remember the exact name"],
            "q10": ["Yes"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["Yes, I have heart failure and an enlarged heart"],
            "q13": ["No"],
            "q14": ["I can get tired more easily than before"],
            "q15": ["No"],
            "q16": ["No"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I do not really drink alcohol"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["A little, because of my heart condition"],
            "q21": ["I have AL amyloidosis, previous blood clot inflammation in the leg, heart failure, an enlarged heart, chronic kidney disease, and abducens palsy"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 18 | DurableKey 8983 | Age 76 | Sex Male | ASA 2
    # ------------------------------------------------------------------
    "patient_18": {
        "name": "Poul",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Poul"],
            "q2": ["I am 76 years old"],
            "q2.1": ["170 cm"],
            "q2.2": ["91.2 kg"],
            "q2.3": ["Male"],

            "q3": ["Yes"],
            "q3b": ["No problems that I know of"],
            "q4": ["No"],
            "q4b": ["No known family reactions"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["No"],
            "q8b": ["No known allergies"],
            "q9": ["Yes"],
            "q9b": ["I take tramadol for knee and back pain"],
            "q10": ["No"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["No"],
            "q13": ["No"],
            "q14": ["No, I can walk up to the second floor without a break"],
            "q15": ["No"],
            "q16": ["No"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I do not really drink alcohol"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["No, I am not especially worried"],
            "q21": ["I have a knee prosthesis, reduced hearing and use hearing aids, and I have knee and back pain treated with tramadol"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 19 | DurableKey 79713 | Age 28 | Sex Female | ASA 3
    # ------------------------------------------------------------------
    "patient_19": {
        "name": "Clara",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Clara"],
            "q2": ["I am 28 years old"],
            "q2.1": ["169 cm"],
            "q2.2": ["57 kg"],
            "q2.3": ["Female"],

            "q3": ["Yes"],
            "q3b": ["No problems that I know of"],
            "q4": ["Uncertain"],
            "q4b": ["I do not know of any family reactions"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["Yes"],
            "q8b": ["I react to morphine, levothyroxine, and infliximab"],
            "q9": ["Yes"],
            "q9b": ["I take lamotrigine and Cymbalta"],
            "q10": ["No"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["No"],
            "q13": ["Yes, I am being checked for lung issues"],
            "q14": ["Yes, I get short of breath on stairs"],
            "q15": ["No"],
            "q16": ["No"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I drink occasionally, but not much"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["A little, mostly because of the lung issues and the examination"],
            "q21": ["I have depression, chronic stomach pain, ulcerative colitis, and I am being checked for possible urinary tract cancer"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 20 | DurableKey 1353055 | Age 35 | Sex Female | ASA 3
    # ------------------------------------------------------------------
    "patient_20": {
        "name": "Julie",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Julie"],
            "q2": ["I am 35 years old"],
            "q2.1": ["165 cm"],
            "q2.2": ["152.8 kg"],
            "q2.3": ["Female"],

            "q3": ["Yes, but mostly for tests and smaller procedures"],
            "q3b": ["No problems that I know of"],
            "q4": ["No"],
            "q4b": ["No known family reactions"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["Yes"],
            "q8b": ["I react to plaster"],
            "q9": ["No"],
            "q9b": ["No daily prescription medication"],
            "q10": ["No"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["No"],
            "q13": ["No"],
            "q14": ["I can get short of breath with longer stairs, mostly because of my weight"],
            "q15": ["No"],
            "q16": ["No"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I drink occasionally"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["Yes, I am a bit nervous about the operation"],
            "q21": ["I have left-sided breast cancer, and I have a high body weight"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 21 | DurableKey 4477 | Age 44 | Sex Female | ASA 3
    # ------------------------------------------------------------------
    "patient_21": {
        "name": "Katrine",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Katrine"],
            "q2": ["I am 44 years old"],
            "q2.1": ["170 cm"],
            "q2.2": ["129 kg"],
            "q2.3": ["Female"],

            "q3": ["Yes"],
            "q3b": ["No problems with anesthesia, but I can be difficult to place an IV in"],
            "q4": ["Uncertain"],
            "q4b": ["I do not know of any family reactions"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["Yes"],
            "q8b": ["I react to nickel"],
            "q9": ["Yes"],
            "q9b": ["I take sertraline"],
            "q10": ["No"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["No"],
            "q13": ["No"],
            "q14": ["I can get short of breath with stairs, mostly because of my weight"],
            "q15": ["No"],
            "q16": ["No"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I drink occasionally, mostly socially"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["Yes, I have anxiety and I am worried about the IV"],
            "q21": ["I have anxiety, type 2 diabetes, diverticular disease, and I am being examined because of possible colon or rectal cancer"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 22 | DurableKey 91578 | Age 55 | Sex Female | ASA 3
    # ------------------------------------------------------------------
    "patient_22": {
        "name": "Mette",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Mette"],
            "q2": ["I am 55 years old"],
            "q2.1": ["163 cm"],
            "q2.2": ["111 kg"],
            "q2.3": ["Female"],

            "q3": ["No"],
            "q3b": ["I have never had general anesthesia before"],
            "q4": ["No"],
            "q4b": ["No known family reactions"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["No"],
            "q8b": ["No known allergies"],
            "q9": ["Yes"],
            "q9b": ["I take Losartan, a PPI, ibuprofen, Gabapentin, and Wegovy"],
            "q10": ["No"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["Yes, I have high blood pressure, but it is well controlled"],
            "q13": ["No"],
            "q14": ["I can get short of breath with stairs, mostly because of my weight and back problems"],
            "q15": ["Yes, I have reflux, but it is well controlled with medication"],
            "q16": ["No"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I drink occasionally"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["Yes, I am a bit nervous because I have not had general anesthesia before"],
            "q21": ["I have high blood pressure, reflux, back problems with two slipped discs, uterine and cervical polyps, and I use Wegovy for weight loss"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 23 | DurableKey 108759 | Age 29 | Sex Male | ASA 3
    # ------------------------------------------------------------------
    "patient_23": {
        "name": "Andreas",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Andreas"],
            "q2": ["I am 29 years old"],
            "q2.1": ["150 cm"],
            "q2.2": ["45.3 kg"],
            "q2.3": ["Male"],

            "q3": ["Yes"],
            "q3b": ["No problems that I know of"],
            "q4": ["Uncertain"],
            "q4b": ["I do not know of any family reactions"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["No"],
            "q8b": ["No known allergies"],
            "q9": ["Yes"],
            "q9b": ["I am not sure what medication is relevant"],
            "q10": ["No"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["No"],
            "q13": ["No"],
            "q14": ["I need help with daily activities because of my disabilities"],
            "q15": ["No"],
            "q16": ["Yes, I have a urinary tract infection"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I do not drink alcohol"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["Yes, I can become anxious or upset in medical situations"],
            "q21": ["I have a severe brain injury, multiple disabilities, developmental disability, behavioural difficulties, ulcerative colitis, and a urinary tract infection"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 24 | DurableKey 2771500 | Age 33 | Sex Male | ASA 3
    # ------------------------------------------------------------------
    "patient_24": {
        "name": "Rasmus",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Rasmus"],
            "q2": ["I am 33 years old"],
            "q2.1": ["185 cm"],
            "q2.2": ["149 kg"],
            "q2.3": ["Male"],

            "q3": ["Yes"],
            "q3b": ["I have had nausea and vomiting after anesthesia before"],
            "q4": ["No"],
            "q4b": ["No known family reactions"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["No"],
            "q8b": ["No known allergies"],
            "q9": ["Yes"],
            "q9b": ["I take oxycodone and Wegovy"],
            "q10": ["No"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["No"],
            "q13": ["No"],
            "q14": ["I can get short of breath with stairs, mostly because of my weight"],
            "q15": ["No"],
            "q16": ["No"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I drink occasionally, mostly socially"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["Yes, mostly because I have vomited after anesthesia before"],
            "q21": ["I have kidney stones with a ureter stone, I take oxycodone, and I have lost weight recently, 12kg on Wegovy"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 25 | DurableKey 13973 | Age 41 | Sex Male | ASA 3
    # ------------------------------------------------------------------
    "patient_25": {
        "name": "Nikolaj",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Nikolaj"],
            "q2": ["I am 41 years old"],
            "q2.1": ["I am not sure"],
            "q2.2": ["I am not sure"],
            "q2.3": ["Male"],

            "q3": ["Yes"],
            "q3b": ["No problems"],
            "q4": ["No"],
            "q4b": ["No known family reactions"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["I have a dental abscess, but no dentures or implants"],

            "q8": ["No"],
            "q8b": ["No known allergies"],
            "q9": ["Yes"],
            "q9b": ["I take pantoprazole"],
            "q10": ["No"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["No"],
            "q13": ["No"],
            "q14": ["No"],
            "q15": ["No, not while I take pantoprazole"],
            "q16": ["Yes, I have an abscess or infection in my mouth"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I drink occasionally"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["A little, mostly because of the infection in my mouth"],
            "q21": ["I have an abscess or infection in my mouth, and I take pantoprazole"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 26 | DurableKey 2341 | Age 56 | Sex Male | ASA 3
    # ------------------------------------------------------------------
    "patient_26": {
        "name": "Søren",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Søren"],
            "q2": ["I am 56 years old"],
            "q2.1": ["177 cm"],
            "q2.2": ["100 kg"],
            "q2.3": ["Male"],

            "q3": ["Yes"],
            "q3b": ["No problems"],
            "q4": ["Uncertain"],
            "q4b": ["I do not know of any family reactions"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["No"],
            "q8b": ["No known allergies"],
            "q9": ["Yes"],
            "q9b": ["I take medication for high blood pressure and type 2 diabetes, but I do not remember all the names"],
            "q10": ["No"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["Yes, I have high blood pressure, but it is well controlled"],
            "q13": ["No"],
            "q14": ["I can get a bit winded on stairs, but nothing dramatic"],
            "q15": ["No"],
            "q16": ["No"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I drink occasionally, but not much"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["Yes, I have PTSD, so hospitals can make me nervous"],
            "q21": ["I have PTSD, high blood pressure, type 2 diabetes, and an umbilical hernia"],
            "q22": ["No additional questions right now"],
        },
    },

    # ------------------------------------------------------------------
    # Patient 27 | DurableKey 26219 | Age 88 | Sex Male | ASA 3
    # ------------------------------------------------------------------
    "patient_27": {
        "name": "Knud",
        "responses": {
            "q0": ["Ready"],
            "q1": ["My name is Knud"],
            "q2": ["I am 88 years old"],
            "q2.1": ["178 cm"],
            "q2.2": ["77.5 kg"],
            "q2.3": ["Male"],

            "q3": ["Yes"],
            "q3b": ["No problems that I know of"],
            "q4": ["No"],
            "q4b": ["No known family reactions"],
            "q5": ["No"],
            "q6": ["No"],
            "q7": ["No"],

            "q8": ["No"],
            "q8b": ["No known allergies"],
            "q9": ["Yes"],
            "q9b": ["I take regular medication, but I am not completely sure of all the names"],
            "q10": ["Uncertain"],
            "q11": ["No"],
            "q11b": ["No supplements"],

            "q12": ["Yes, I have atrial fibrillation and a pacemaker"],
            "q13": ["No"],
            "q14": ["I can get tired more easily because of my age and heart condition"],
            "q15": ["No"],
            "q16": ["No"],

            "q17": ["No, I do not smoke"],
            "q17b": ["I do not use nicotine"],
            "q18": ["I do not really drink alcohol"],

            "q19": ["No"],
            "q19b": ["No"],
            "q20": ["No, I feel okay about it"],
            "q21": ["I have colon cancer, atrial fibrillation, a pacemaker, a colostomy, and previous prostate cancer"],
            "q22": ["No additional questions right now"],
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

        # Track acknowledgment vs bare question responses
        self._acknowledged_count = 0
        self._bare_question_count = 0

        # Track which follow-up questions were triggered
        self._followup_ids = {"q3b", "q4b", "q8b", "q9b", "q11b", "q17b", "q19b"}
        self._followups_triggered: List[str] = []

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

        print(f"\n{C.BOLD}{'=' * 60}")
        print(f"  Persona: {self.persona['name']}  |  Run {self.run_number}")
        print(f"  Conversation ID: {self.conversation_id}")
        print(f"  API: {self.api}")
        print(f"{'=' * 60}{C.RESET}\n")

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
                    print(
                        f"  {C.RED}⚠ Loop detected on {current_qid} ({attempts} attempts) — forcing bare answer{C.RESET}")
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
                bot_text, done, current_qid, was_acknowledged = self._send_message(response)

                # Track acknowledgment from API
                if not done:
                    if was_acknowledged:
                        self._acknowledged_count += 1
                    else:
                        self._bare_question_count += 1

                # Track follow-up triggers
                if current_qid and current_qid in self._followup_ids and current_qid not in self._followups_triggered:
                    self._followups_triggered.append(current_qid)

                bot_meta = f"asking: {current_qid}" if current_qid else "done"
                if not done:
                    bot_meta += f" | {'ack' if was_acknowledged else 'bare'}"
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

            # Save consultation to disk
            try:
                self._save_consultation(summary)
            except Exception as e:
                self._error(f"Failed to save consultation: {e}")

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

    def _send_message(self, message: str) -> Tuple[str, bool, Optional[str], bool]:
        r = requests.post(
            f"{self.api}/conversations/{self.conversation_id}/message",
            json={"message": message},
            timeout=300,
        )
        r.raise_for_status()
        data = r.json()
        return (
            data["bot_text"],
            data.get("done", False),
            data.get("current_question_id"),
            data.get("acknowledged", False),
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
            timeout=300,
        )
        r.raise_for_status()
        return r.json()

    def _delete_conversation(self):
        requests.delete(
            f"{self.api}/conversations/{self.conversation_id}",
            timeout=10,
        )

    def _save_consultation(self, summary: dict):
        requests.post(
            f"{self.api}/consultations/save",
            json=summary,
            timeout=30,
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

        total_bot_responses = self._acknowledged_count + self._bare_question_count
        ack_rate = (self._acknowledged_count / total_bot_responses * 100) if total_bot_responses > 0 else 0
        lines.append(f"Acknowledgment rate: {self._acknowledged_count}/{total_bot_responses} ({ack_rate:.0f}%)")
        lines.append(f"Bare question responses: {self._bare_question_count}/{total_bot_responses}")

        if re_asked:
            lines.append(f"Re-asked questions:")
            for qid, attempts in re_asked.items():
                lines.append(f"  {qid}: {attempts} attempts")

        lines.append("")
        fu_str = ", ".join(self._followups_triggered) if self._followups_triggered else "none"
        lines.append(f"Follow-ups triggered: {len(self._followups_triggered)}/7 ({fu_str})")

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
        print(f"\n{C.BOLD}{'─' * 60}")
        print(f"  SUMMARY")
        print(f"{'─' * 60}{C.RESET}")
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

        total_bot_responses = self._acknowledged_count + self._bare_question_count
        ack_rate = (self._acknowledged_count / total_bot_responses * 100) if total_bot_responses > 0 else 0

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
            "acknowledged": self._acknowledged_count,
            "bare_question": self._bare_question_count,
            "acknowledgment_rate": round(ack_rate, 1),
            "followups_triggered": list(self._followups_triggered),
            "followup_count": len(self._followups_triggered),
        }

        # Format duration
        minutes, secs = divmod(int(elapsed), 60)
        time_str = f"{minutes}m {secs}s" if minutes else f"{secs}s"

        fu_str = ", ".join(self._followups_triggered) if self._followups_triggered else "none"

        print(f"\n{C.BOLD}{'=' * 60}")
        print(f"  RESULTS — {self.persona['name']} Run {self.run_number}")
        print(f"{'=' * 60}{C.RESET}")
        print(f"  Total turns:              {self.turn_count}")
        print(f"  Questions asked:          {total_questions}")
        print(f"  First-attempt accepted:   {first_attempt}/{total_questions} ({first_attempt_rate:.0f}%)")
        print(f"  Re-asked questions:       {len(re_asked)}")
        print(f"  Follow-ups triggered:     {len(self._followups_triggered)}/7 ({fu_str})")
        print(f"  Acknowledgments:          {self._acknowledged_count}/{total_bot_responses} ({ack_rate:.0f}%)")
        print(f"  Bare questions:           {self._bare_question_count}/{total_bot_responses}")
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

        ack_rates = [r.get("acknowledgment_rate", 0) for r in runs]
        avg_ack = sum(ack_rates) / len(ack_rates) if ack_rates else 0

        fu_counts = [r.get("followup_count", 0) for r in runs]
        avg_fu = sum(fu_counts) / len(fu_counts) if fu_counts else 0

        out_fn(f"  {'─' * 56}")
        out_fn(f"  {C.BOLD}{persona_key.upper()}{C.RESET} ({PERSONAS[persona_key]['name']})")
        out_fn(f"  {'─' * 56}")
        out_fn(f"  Completed:              {passed}/{total}")
        out_fn(f"  Avg turns:              {avg_turns:.1f}  (min {min_turns}, max {max_turns_val})")
        out_fn(f"  Avg first-attempt rate: {avg_fa:.1f}%")
        out_fn(f"  Avg acknowledgment rate:{avg_ack:.1f}%")
        out_fn(f"  Avg follow-ups triggered:{avg_fu:.1f}/7")
        out_fn(f"  Avg duration:           {avg_time_str}  (total {total_time_str})")

        # Per-run table
        out_fn()
        out_fn(
            f"  {'Run':<6} {'Turns':<8} {'1st-att':<18} {'Ack':<7} {'F-ups':<7} {'Re-asked':<22} {'Time':<8} {'Result'}")
        out_fn(
            f"  {'───':<6} {'─────':<8} {'───────':<18} {'───':<7} {'─────':<7} {'────────':<22} {'────':<8} {'──────'}")

        for r in runs:
            re_asked_str = ", ".join(f"{q}({a}x)" for q, a in r["re_asked"].items()) if r["re_asked"] else "—"
            status = f"{C.GREEN}PASS{C.RESET}" if r["passed"] else f"{C.RED}FAIL{C.RESET}"
            fa_str = f"{r['first_attempt_accepted']}/{r['questions_asked']} ({r['first_attempt_rate']}%)"
            ack_str = f"{r.get('acknowledgment_rate', 0):.0f}%"
            fu_str = f"{r.get('followup_count', 0)}/7"
            elapsed = r.get("elapsed_s", 0)
            m, s = divmod(int(elapsed), 60)
            time_str = f"{m}m{s:02d}s" if m else f"{s}s"
            out_fn(
                f"  {r['run']:<6} {r['total_turns']:<8} {fa_str:<18} {ack_str:<7} {fu_str:<7} {re_asked_str:<22} {time_str:<8} {status}")

        # Which questions get re-asked most often
        reask_totals: Dict[str, int] = {}
        for r in runs:
            for qid, attempts in r["re_asked"].items():
                reask_totals[qid] = reask_totals.get(qid, 0) + 1

        if reask_totals:
            out_fn()
            out_fn(f"  Questions re-asked (across {total} runs):")
            for qid, count in sorted(reask_totals.items(), key=lambda x: -x[1]):
                out_fn(f"    {qid}: re-asked in {count}/{total} runs ({count / total * 100:.0f}%)")

        # Follow-up trigger consistency
        all_followup_ids = {"q3b", "q4b", "q8b", "q9b", "q11b", "q17b", "q19b"}
        fu_trigger_counts: Dict[str, int] = {}
        for r in runs:
            for fid in r.get("followups_triggered", []):
                fu_trigger_counts[fid] = fu_trigger_counts.get(fid, 0) + 1

        out_fn()
        out_fn(f"  Follow-up triggers (across {total} runs):")
        for fid in sorted(all_followup_ids):
            count = fu_trigger_counts.get(fid, 0)
            if count > 0:
                consistency = f"{count}/{total} runs ({count / total * 100:.0f}%)"
            else:
                consistency = "never triggered"
            out_fn(f"    {fid}: {consistency}")

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
        print(f"\n{'=' * 60}")
        print(f"  PERSONA SUMMARY — {persona_key.upper()}")
        print(f"{'=' * 60}")
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

    out(f"\n{'=' * 60}")
    out(f"  FULL EVALUATION SUMMARY")
    out(f"{'=' * 60}")
    out(f"  Date:             {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    out(f"  Runs per persona: {args.runs}")
    out(f"  Transcripts dir:  {TRANSCRIPT_DIR}")
    out()

    # ── Per-persona detail (repeated in saved file) ──
    for persona_key in personas_to_run:
        print_persona_summary(persona_key, all_stats[persona_key], out_fn=out)

    # ── Cross-persona comparison table ──
    out(f"  {'=' * 56}")
    out(f"  CROSS-PERSONA COMPARISON")
    out(f"  {'=' * 56}")
    out()
    out(f"  {'Persona':<18} {'Completed':<12} {'Avg turns':<12} {'Avg 1st-att':<14} {'Avg ack':<10} {'Avg f-ups':<10} {'Avg time':<10} {'Most re-asked'}")
    out(f"  {'───────':<18} {'─────────':<12} {'─────────':<12} {'───────────':<14} {'───────':<10} {'─────────':<10} {'────────':<10} {'─────────────'}")

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
        ack_rates = [r.get("acknowledgment_rate", 0) for r in runs]
        fu_counts = [r.get("followup_count", 0) for r in runs]
        elapsed_list = [r.get("elapsed_s", 0) for r in runs]
        avg_turns = sum(turns_list) / len(turns_list) if turns_list else 0
        avg_fa = sum(fa_rates) / len(fa_rates) if fa_rates else 0
        avg_ack = sum(ack_rates) / len(ack_rates) if ack_rates else 0
        avg_fu = sum(fu_counts) / len(fu_counts) if fu_counts else 0
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

        out(f"  {persona_key:<18} {passed}/{total:<10} {avg_turns:<12.1f} {avg_fa:<14.1f}% {avg_ack:<9.1f}% {avg_fu:<9.1f} {avg_time_str:<10} {worst_str}")

    out()
    out(f"  {'─' * 56}")
    overall_rate = (total_passed / total_runs * 100) if total_runs > 0 else 0
    grand_m, grand_s = divmod(int(grand_total_elapsed), 60)
    grand_time_str = f"{grand_m}m {grand_s}s" if grand_m else f"{grand_s}s"
    out(f"  OVERALL: {total_passed}/{total_runs} completed ({overall_rate:.0f}%)  —  Total time: {grand_time_str}")
    out(f"  {'─' * 56}")
    out()

    # Write summary file
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"  {C.GRAY}📄 Summary saved: {summary_path}{C.RESET}\n")

    sys.exit(0 if total_passed == total_runs else 1)


if __name__ == "__main__":
    main()