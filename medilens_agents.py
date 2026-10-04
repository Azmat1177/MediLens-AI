"""
MediLens AI - Multi-agent system (LangGraph)

Agents:
  0 Orchestrator   1 Extraction   2 Analysis   3 Medical RAG
  4 Verification   5 Explanation  6 Medicine   7 Safety Guardrail (input + output)

Run demo (no API keys needed):   python medilens_agents.py
Install:                         pip install langgraph pydantic
Optional (Gemini LLM/vision/OCR): pip install google-genai  and set GEMINI_API_KEY

Design rules
  - Numeric comparison (high/low/critical) is deterministic Python, never LLM.
  - LLM is only used to polish wording of already-verified content.
  - Human-in-the-loop: when input is unclear, the graph ends with a question in
    `final_answer` and `needs_user` set. The caller re-invokes with the user's reply.
"""
from __future__ import annotations

import os
import re
from typing import TypedDict

from langgraph.graph import END, StateGraph

MAX_RETRIES = 2
OCR_MIN_CONF = 0.6
MED_MIN_CONF = 0.7
DISCLAIMER = {
    "en": "This is not medical advice. Please consult your doctor or pharmacist.",
    "roman_urdu": "Ye medical advice nahi hai. Meherbani karke apne doctor ya pharmacist se mashwara karein.",
}


# --------------------------------------------------------------------------
# Shared state: every agent reads it and writes only its own keys
# --------------------------------------------------------------------------
class State(TypedDict, total=False):
    language: str            # "en" | "roman_urdu"
    user_text: str           # free-text question from user
    file_path: str           # uploaded report / medicine image
    raw_text: str            # OCR text (injected for demo / tests)
    medicine_hint: dict      # vision output (injected for demo / tests)
    patient: dict            # {"sex": "F", "age": 30}
    request_type: str        # "report" | "medicine"
    blocked: bool
    emergency: bool
    needs_user: str          # question to ask the user (human-in-the-loop)
    extracted: list
    ocr_confidence: float
    findings: list
    evidence: dict           # finding key -> list of chunks
    medicine: dict
    verification: dict
    retry_count: int
    feedback: str            # "analysis" | "rag" | ""
    escalate: bool
    final_answer: str
    sources: list
    trace: list


def _trace(state: State, msg: str) -> list:
    return state.get("trace", []) + [msg]


# --------------------------------------------------------------------------
# Knowledge (replace with Qdrant / Chroma collections in production)
# --------------------------------------------------------------------------
REFERENCE = {  # adult ranges: (low, high, critical_low, critical_high, unit)
    "hemoglobin": {"M": (13.5, 17.5, 7.0, 20.0, "g/dL"), "F": (12.0, 15.5, 7.0, 20.0, "g/dL")},
    "wbc": {"*": (4.0, 11.0, 2.0, 30.0, "x10^3/uL")},
    "platelets": {"*": (150, 450, 50, 1000, "x10^3/uL")},
    "mcv": {"*": (80, 100, 60, 130, "fL")},
}
ALIASES = {
    "hemoglobin": "hemoglobin", "hb": "hemoglobin", "haemoglobin": "hemoglobin",
    "wbc": "wbc", "white blood cells": "wbc", "tlc": "wbc",
    "platelets": "platelets", "plt": "platelets", "platelet count": "platelets",
    "mcv": "mcv",
}

LAB_KB = [  # (id, finding key, status, text, source)
    ("lab-001", "hemoglobin", "low", "Low hemoglobin means fewer oxygen-carrying red cells (anemia). Common causes include iron deficiency, blood loss and chronic disease. A doctor can find the cause with more tests.", "MedlinePlus"),
    ("lab-002", "hemoglobin", "high", "High hemoglobin can be linked to dehydration, smoking or living at high altitude. A doctor should review persistent high values.", "MedlinePlus"),
    ("lab-003", "mcv", "low", "A low MCV means red cells are smaller than usual. It often goes with iron deficiency.", "MedlinePlus"),
    ("lab-004", "wbc", "high", "A high white cell count often shows the body is fighting infection or inflammation.", "MedlinePlus"),
    ("lab-005", "wbc", "low", "A low white cell count can follow viral infections or some medicines. Persistent low values need a doctor's review.", "MedlinePlus"),
    ("lab-006", "platelets", "low", "Low platelets can make bruising or bleeding easier. A doctor should check the cause.", "MedlinePlus"),
    ("lab-007", "platelets", "high", "High platelets can occur with infection, inflammation or iron deficiency.", "MedlinePlus"),
]
DRUG_KB = {  # name -> record (RxNorm/DRAP/DailyMed in production)
    "panadol": {"ingredient": "Paracetamol (Acetaminophen)", "strength": "500mg",
                "use": "Pain relief and fever reduction.",
                "same_formula": ["Calpol", "Disprol", "Paracetamol (generic)"], "source": "DailyMed / DRAP"},
    "brufen": {"ingredient": "Ibuprofen", "strength": "400mg",
               "use": "Pain, inflammation and fever.",
               "same_formula": ["Advil", "Ibuprofen (generic)"], "source": "DailyMed / DRAP"},
}


def retrieve(key: str, status: str, k: int = 2) -> list:
    """Stand-in for hybrid BM25 + vector search + reranker."""
    hits = [c for c in LAB_KB if c[1] == key and c[2] == status]
    return [{"id": c[0], "text": c[3], "source": c[4]} for c in hits[:k]]


# --------------------------------------------------------------------------
# Optional Gemini (polish wording, read images). Never decides medical facts.
# Key: set env var GEMINI_API_KEY (free key from https://aistudio.google.com/apikey)
# Model: set env var MEDILENS_MODEL to override the default.
# --------------------------------------------------------------------------
import json
import mimetypes

GEMINI_MODEL = os.getenv("MEDILENS_MODEL", "gemini-2.5-flash")


def _gemini():
    """Return (client, types) or (None, None) if no key / SDK."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return None, None
    try:
        from google import genai
        from google.genai import types
        return genai.Client(api_key=key), types
    except Exception:
        return None, None


def _image_part(types, file_path: str):
    mime = mimetypes.guess_type(file_path)[0] or "image/jpeg"
    with open(file_path, "rb") as f:
        return types.Part.from_bytes(data=f.read(), mime_type=mime)


def llm_polish(text: str, language: str) -> str:
    client, types = _gemini()
    if not client:
        return text
    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=("Rewrite the text in simple, kind language. Do NOT add, remove or change "
                                    "any medical fact, number or recommendation. Do not give doses or diagnoses. "
                                    f"Output language: {language}."),
                temperature=0.2,
            ),
        )
        return resp.text or text
    except Exception:
        return text  # fail safe: keep deterministic text


def run_ocr(file_path: str) -> str:
    """Report image/PDF -> text. Gemini vision first, pytesseract as fallback."""
    client, types = _gemini()
    if client:
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[_image_part(types, file_path),
                          "Transcribe this lab report. One test per line exactly as: "
                          "Name value unit (low-high). No extra text."],
            )
            return resp.text or ""
        except Exception:
            pass
    try:
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(file_path))
    except Exception:
        return ""


def run_vision(file_path: str) -> dict:
    """Medicine photo -> {name, strength, confidence}."""
    empty = {"name": "", "strength": "", "confidence": 0.0}
    client, types = _gemini()
    if not client or not file_path:
        return empty
    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[_image_part(types, file_path), "Identify this medicine."],
            config=types.GenerateContentConfig(
                system_instruction=('Read the medicine package. Reply with JSON only: '
                                    '{"name": "", "strength": "", "confidence": 0.0}. '
                                    'Use confidence below 0.7 if text is blurry or you are guessing.'),
                response_mime_type="application/json",
                temperature=0,
            ),
        )
        data = json.loads(resp.text)
        return {"name": str(data.get("name", "")), "strength": str(data.get("strength", "")),
                "confidence": float(data.get("confidence", 0.0))}
    except Exception:
        return empty


# --------------------------------------------------------------------------
# Agent 7: Safety guardrail (input + output)
# --------------------------------------------------------------------------
RED_FLAGS = re.compile(r"chest pain|can'?t breathe|trouble breathing|suicid|unconscious|severe bleeding|seene mein dard|saans nahi", re.I)
BLOCK_INPUT = re.compile(r"(change|increase|decrease|double).*(dose|dosage)|prescribe|stop taking|dose kitni|dawai band", re.I)
UNSAFE_OUTPUT = re.compile(r"\b(take|start|stop|increase|decrease|double|skip)\b[^.]{0,40}\b(mg|tablet|tablets|dose|dosage|medicine|medication)\b", re.I)
DIAGNOSIS = re.compile(r"\byou (have|are suffering from) (cancer|leukemia|diabetes|anemia)\b", re.I)


def guard_in(state: State) -> dict:
    text = state.get("user_text", "")
    if RED_FLAGS.search(text):
        return {"emergency": True, "final_answer": "This may be an emergency. Please call your local emergency number or go to the nearest hospital now.",
                "trace": _trace(state, "guard_in: emergency")}
    if BLOCK_INPUT.search(text):
        return {"blocked": True, "final_answer": "I can't prescribe or change doses. Please ask your doctor or pharmacist. I can still explain your report or what a medicine is generally used for.",
                "trace": _trace(state, "guard_in: blocked")}
    return {"trace": _trace(state, "guard_in: ok")}


def guard_out(state: State) -> dict:
    lang = state.get("language", "en")
    answer = state.get("final_answer", "")
    removed, kept_lines = 0, []
    for line in answer.split("\n"):  # keep line breaks, filter sentence by sentence
        parts = re.split(r"(?<=[.!?])\s+", line)
        safe = [s for s in parts if not UNSAFE_OUTPUT.search(s) and not DIAGNOSIS.search(s)]
        removed += len(parts) - len(safe)
        kept_lines.append(" ".join(safe))
    answer = "\n".join(kept_lines)
    if state.get("sources"):
        answer += "\n\nSources: " + ", ".join(sorted(set(state["sources"])))
    answer += "\n\n" + DISCLAIMER.get(lang, DISCLAIMER["en"])
    return {"final_answer": answer, "trace": _trace(state, f"guard_out: removed {removed} unsafe sentence(s)")}


# --------------------------------------------------------------------------
# Agent 0: Orchestrator
# --------------------------------------------------------------------------
def orchestrator(state: State) -> dict:
    is_med = bool(state.get("medicine_hint")) or (
        state.get("file_path", "").lower().endswith((".jpg", ".jpeg", ".png")) and not state.get("raw_text"))
    has_report = bool(state.get("raw_text") or state.get("file_path"))
    if is_med:
        rtype = "medicine"
    elif has_report:
        rtype = "report"
    elif state.get("user_text"):
        rtype = "general"
    else:
        rtype = "report"
    return {"request_type": rtype, "retry_count": state.get("retry_count", 0),
            "trace": _trace(state, f"orchestrator: {rtype}")}


# --------------------------------------------------------------------------
# General question agent (terms, tests, "what does X mean")
# --------------------------------------------------------------------------
def general_agent(state: State) -> dict:
    lang = state.get("language", "en")
    client, types = _gemini()
    if not client:
        msg = ("I can explain general medical terms when a Gemini key is set. "
               "Meanwhile, upload a report or a medicine photo, or paste report text, and I will analyze it.")
        return {"final_answer": msg, "trace": _trace(state, "general: no LLM key")}
    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=state["user_text"],
            config=types.GenerateContentConfig(
                system_instruction=("You explain medical terms and lab tests in simple, short language. "
                                    "Never diagnose, never give doses, never tell the user to start, stop or "
                                    "change a medicine. For symptoms or decisions, advise seeing a doctor. "
                                    f"Answer in: {lang}."),
                temperature=0.3,
            ),
        )
        return {"final_answer": resp.text or "", "trace": _trace(state, "general: answered")}
    except Exception:
        return {"final_answer": "Sorry, I could not answer right now. Please try again.",
                "trace": _trace(state, "general: error")}


# --------------------------------------------------------------------------
# Agent 1: Report extraction
# --------------------------------------------------------------------------
LINE = re.compile(r"^\s*(?P<name>[A-Za-z][A-Za-z ]+?)\s+(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z/%^0-9.]+)?\s*(?:\((?P<lo>\d+(?:\.\d+)?)\s*-\s*(?P<hi>\d+(?:\.\d+)?)\))?\s*$")


def extraction_agent(state: State) -> dict:
    text = state.get("raw_text") or (run_ocr(state["file_path"]) if state.get("file_path") else "")
    lines = [l for l in text.splitlines() if l.strip()]
    rows = []
    for l in lines:
        m = LINE.match(l)
        if not m:
            continue
        key = ALIASES.get(m["name"].strip().lower())
        if key:
            rows.append({"key": key, "value": float(m["value"]), "unit": m["unit"] or "",
                         "range": (float(m["lo"]), float(m["hi"])) if m["lo"] else None})
    conf = len(rows) / len(lines) if lines else 0.0
    out = {"extracted": rows, "ocr_confidence": round(conf, 2)}
    if conf < OCR_MIN_CONF or not rows:
        out["needs_user"] = "ocr"
        out["final_answer"] = "I could not read your report clearly. Please upload a sharper photo or PDF, or type the values."
    out["trace"] = _trace(state, f"extraction: {len(rows)} values, conf={conf:.2f}")
    return out


# --------------------------------------------------------------------------
# Agent 2: Analysis (deterministic)
# --------------------------------------------------------------------------
def classify(key: str, value: float, report_range, sex: str):
    ref = REFERENCE.get(key, {})
    lo_hi = ref.get(sex) or ref.get("*")
    crit_lo, crit_hi = (lo_hi[2], lo_hi[3]) if lo_hi else (None, None)
    rng = report_range or (lo_hi[0], lo_hi[1] if lo_hi else None) if (report_range or lo_hi) else None
    if not rng or rng[0] is None:
        return "unknown", "unknown"
    lo, hi = rng
    if crit_lo is not None and value < crit_lo or crit_hi is not None and value > crit_hi:
        return ("low" if value < lo else "high"), "critical"
    if value < lo:
        return "low", "abnormal"
    if value > hi:
        return "high", "abnormal"
    return "normal", "normal"


def analysis_agent(state: State) -> dict:
    sex = state.get("patient", {}).get("sex", "*")
    findings = []
    for r in state.get("extracted", []):
        status, severity = classify(r["key"], r["value"], r["range"], sex)
        findings.append({**r, "status": status, "severity": severity})
    keys = {(f["key"], f["status"]) for f in findings}
    patterns = []
    if ("hemoglobin", "low") in keys and ("mcv", "low") in keys:
        patterns.append("Low hemoglobin with low MCV (small red cells) often fits an iron-related pattern.")
    return {"findings": findings, "verification": {"patterns": patterns}, "feedback": "",
            "trace": _trace(state, f"analysis: {sum(f['severity'] != 'normal' for f in findings)} abnormal")}


# --------------------------------------------------------------------------
# Agent 3: Medical RAG
# --------------------------------------------------------------------------
def rag_agent(state: State) -> dict:
    evidence = {}
    for f in state.get("findings", []):
        if f["severity"] != "normal" and f["status"] != "unknown":
            evidence[f["key"]] = retrieve(f["key"], f["status"])
    return {"evidence": evidence, "feedback": "", "trace": _trace(state, f"rag: {len(evidence)} findings with evidence")}


# --------------------------------------------------------------------------
# Agent 6: Medicine agent (+ drug RAG)
# --------------------------------------------------------------------------
def medicine_agent(state: State) -> dict:
    hint = state.get("medicine_hint") or run_vision(state.get("file_path", ""))
    name = hint.get("name", "").strip().lower()
    conf = hint.get("confidence", 0.0)
    if conf < MED_MIN_CONF or name not in DRUG_KB:
        guess = hint.get("name") or "this medicine"
        return {"needs_user": "medicine", "final_answer": f"I am not sure about the image. Is this {guess} {hint.get('strength', '')}? Please confirm the name and strength.",
                "trace": _trace(state, f"medicine: unclear (conf={conf})")}
    rec = DRUG_KB[name]
    return {"medicine": {"name": name.title(), **rec}, "trace": _trace(state, f"medicine: matched {name}")}


# --------------------------------------------------------------------------
# Agent 4: Verification
# --------------------------------------------------------------------------
def verification_agent(state: State) -> dict:
    retries = state.get("retry_count", 0)
    issues, feedback = [], ""
    patterns = state.get("verification", {}).get("patterns", [])

    if state["request_type"] == "medicine":
        rec = state.get("medicine", {})
        if not rec.get("ingredient"):
            issues.append("no drug record")
        if UNSAFE_OUTPUT.search(rec.get("use", "")):
            issues.append("unsafe drug text")
        verdict = "pass" if not issues else "escalate"
        return {"verification": {"verdict": verdict, "issues": issues}, "escalate": verdict == "escalate",
                "trace": _trace(state, f"verification(medicine): {verdict}")}

    findings = state.get("findings", [])
    sex = state.get("patient", {}).get("sex", "*")
    # 1) rule re-check (independent recomputation)
    for f in findings:
        status, severity = classify(f["key"], f["value"], f["range"], sex)
        if (status, severity) != (f["status"], f["severity"]):
            issues.append(f"rule mismatch: {f['key']}")
            feedback = "analysis"
    # 2) groundedness: every abnormal finding needs retrieved evidence
    for f in findings:
        if f["severity"] != "normal" and f["status"] != "unknown" and not state.get("evidence", {}).get(f["key"]):
            issues.append(f"no evidence: {f['key']}")
            feedback = feedback or "rag"
    # 3) safety policy on all text we may show
    all_text = [c["text"] for chunks in state.get("evidence", {}).values() for c in chunks] + patterns
    if any(UNSAFE_OUTPUT.search(t) or DIAGNOSIS.search(t) for t in all_text):
        issues.append("unsafe wording in evidence")
        feedback = feedback or "rag"
    # 4) decision
    critical = any(f["severity"] == "critical" for f in findings)
    if critical:
        verdict = "escalate"
    elif issues and retries < MAX_RETRIES:
        verdict = "revise"
    elif issues:
        verdict = "escalate"
    else:
        verdict = "pass"
    return {"verification": {"verdict": verdict, "issues": issues, "patterns": patterns},
            "feedback": feedback, "escalate": verdict == "escalate",
            "retry_count": retries + (1 if verdict == "revise" else 0),
            "trace": _trace(state, f"verification: {verdict} {issues}")}


# --------------------------------------------------------------------------
# Agent 5: Explanation
# --------------------------------------------------------------------------
T = {
    "en": {"low": "is lower than normal", "high": "is higher than normal", "normal": "is in the normal range",
           "head": "Here is a simple explanation of your report:", "urgent": "Some values look seriously abnormal. Please contact a doctor urgently.",
           "pat": "Pattern noticed:", "rx_head": "About this medicine:", "ing": "Active ingredient", "use": "General use",
           "brands": "Other brands with the same ingredient and strength", "ask": "Please confirm any medicine decision with your doctor or pharmacist."},
    "roman_urdu": {"low": "normal se kam hai", "high": "normal se zyada hai", "normal": "normal range mein hai",
                   "head": "Aap ki report ki aasan wazahat:", "urgent": "Kuch values bohat ghair-mamooli hain. Meherbani karke foran doctor se rabta karein.",
                   "pat": "Nazar aane wala pattern:", "rx_head": "Is dawai ke baare mein:", "ing": "Active ingredient", "use": "Aam istemal",
                   "brands": "Isi ingredient aur strength wali dusri brands", "ask": "Dawai ke baare mein koi bhi faisla doctor ya pharmacist se confirm karke karein."},
}


def explanation_agent(state: State) -> dict:
    lang = state.get("language", "en")
    t = T.get(lang, T["en"])
    sources, lines = [], []
    if state["request_type"] == "medicine":
        m = state["medicine"]
        lines = [t["rx_head"], f"{m['name']} {m['strength']}", f"{t['ing']}: {m['ingredient']}", f"{t['use']}: {m['use']}",
                 f"{t['brands']}: {', '.join(m['same_formula'])}", t["ask"]]
        sources.append(m["source"])
    else:
        lines.append(t["head"])
        for f in state["findings"]:
            lines.append(f"- {f['key'].upper()} {f['value']} {f['unit']} {t[f['status']] if f['status'] in t else ''}".rstrip())
            for c in state.get("evidence", {}).get(f["key"], []):
                lines.append(f"  {c['text']}")
                sources.append(c["source"])
        for p in state.get("verification", {}).get("patterns", []):
            lines.append(f"{t['pat']} {p}")
        lines.append(t["ask"])
    return {"final_answer": llm_polish("\n".join(lines), lang), "sources": sources, "trace": _trace(state, "explanation: done")}


def escalate_node(state: State) -> dict:
    lang = state.get("language", "en")
    t = T.get(lang, T["en"])
    body = [t["urgent"]]
    if state["request_type"] == "report":
        for f in state.get("findings", []):
            if f["severity"] != "normal":
                body.append(f"- {f['key'].upper()} {f['value']} {f['unit']} {t.get(f['status'], '')}")
    return {"final_answer": "\n".join(body), "trace": _trace(state, "escalate: urgent message")}


# --------------------------------------------------------------------------
# Routing + graph
# --------------------------------------------------------------------------
def route_after_guard_in(s: State) -> str:
    return "guard_out" if s.get("blocked") or s.get("emergency") else "orchestrator"


def route_after_orch(s: State) -> str:
    return {"medicine": "medicine_agent", "general": "general_agent"}.get(s["request_type"], "extraction_agent")


def route_ask_user(next_node: str):
    return lambda s: "guard_out" if s.get("needs_user") else next_node


def route_after_verification(s: State) -> str:
    v = s["verification"]["verdict"]
    if v == "pass":
        return "explanation_agent"
    if v == "revise":
        return "analysis_agent" if s.get("feedback") == "analysis" else "rag_agent"
    return "escalate_node"


def build_graph():
    g = StateGraph(State)
    for name, fn in [("guard_in", guard_in), ("orchestrator", orchestrator), ("extraction_agent", extraction_agent),
                     ("analysis_agent", analysis_agent), ("rag_agent", rag_agent), ("medicine_agent", medicine_agent),
                     ("verification_agent", verification_agent), ("explanation_agent", explanation_agent),
                     ("escalate_node", escalate_node), ("general_agent", general_agent), ("guard_out", guard_out)]:
        g.add_node(name, fn)
    g.set_entry_point("guard_in")
    g.add_conditional_edges("guard_in", route_after_guard_in, ["guard_out", "orchestrator"])
    g.add_conditional_edges("orchestrator", route_after_orch, ["medicine_agent", "extraction_agent", "general_agent"])
    g.add_edge("general_agent", "guard_out")
    g.add_conditional_edges("extraction_agent", route_ask_user("analysis_agent"), ["guard_out", "analysis_agent"])
    g.add_edge("analysis_agent", "rag_agent")
    g.add_edge("rag_agent", "verification_agent")
    g.add_conditional_edges("medicine_agent", route_ask_user("verification_agent"), ["guard_out", "verification_agent"])
    g.add_conditional_edges("verification_agent", route_after_verification,
                            ["explanation_agent", "analysis_agent", "rag_agent", "escalate_node"])
    g.add_edge("explanation_agent", "guard_out")
    g.add_edge("escalate_node", "guard_out")
    g.add_edge("guard_out", END)
    return g.compile()


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_graph()

    cbc = """Hemoglobin 9.8 g/dL (12.0-15.5)
MCV 72 fL (80-100)
WBC 7.2 x10^3/uL (4.0-11.0)
Platelets 250 x10^3/uL (150-450)"""

    demos = {
        "1) CBC report (English)": {"raw_text": cbc, "patient": {"sex": "F", "age": 30}, "language": "en"},
        "2) CBC report (Roman Urdu)": {"raw_text": cbc, "patient": {"sex": "F"}, "language": "roman_urdu"},
        "3) Critical value": {"raw_text": "Hemoglobin 5.2 g/dL (12.0-15.5)", "patient": {"sex": "F"}, "language": "en"},
        "4) Medicine, clear image": {"medicine_hint": {"name": "Panadol", "strength": "500mg", "confidence": 0.93}, "language": "en"},
        "5) Medicine, unclear image": {"medicine_hint": {"name": "Panadol", "strength": "500mg", "confidence": 0.4}, "language": "en"},
        "6) Unsafe request": {"user_text": "Can you change my dose of metformin?", "language": "en"},
        "7) Emergency": {"user_text": "I have chest pain", "language": "en"},
    }
    for title, inp in demos.items():
        out = app.invoke(inp)
        print("=" * 70, f"\n{title}\n" + "-" * 70)
        print(out["final_answer"])
        print("\ntrace:", " > ".join(out["trace"]))
