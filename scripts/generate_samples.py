"""Generates synthetic (fictional) HR documents + an answer key for testing the legal RAG agent.
Run:  python scripts/generate_samples.py
All companies, people and numbers are invented. The answer key is a draft: verify every
expected section against India Code before using it in your dissertation.
"""
import json
from pathlib import Path

OUT = Path("data/samples")
OUT.mkdir(parents=True, exist_ok=True)
BANNER = "SYNTHETIC SAMPLE - FICTIONAL DOCUMENT FOR SOFTWARE TESTING ONLY"


def make(title, intro, clauses):
    lines = [BANNER, "", title.upper(), "", intro, ""]
    for i, (head, body) in enumerate(clauses, 1):
        lines += [f"{i}. {head}", body, ""]
    return "\n".join(lines)


DOCS = []

# 1. Clean agreement: negative control (nothing should be flagged as a legal problem)
DOCS.append(dict(
    id="doc01", file="01_clean_agreement.txt", fmt="txt",
    text=make("Employment Agreement",
              "Between Northwind Software Pvt. Ltd. (Company) and Priya Nair (Employee), Software Engineer, Bengaluru.",
              [("Probation", "Probation is 3 months. Either party may end employment with 15 days' written notice during probation."),
               ("Notice Period", "After confirmation, either party shall give 60 days' written notice or salary in lieu of notice."),
               ("Confidentiality", "The Employee shall keep Company confidential information secret during employment and for 2 years afterwards."),
               ("Final Settlement", "All dues will be paid within two working days of the last working day, as required by applicable law."),
               ("Benefits", "Provident fund, gratuity and maternity benefits will be provided as per applicable law."),
               ("Working Hours", "Standard hours are 8 hours a day, 5 days a week. Overtime is paid as per applicable law.")]),
    truth=dict(issues=[], should_not_flag=["60-day notice period", "confidentiality clause", "final settlement within two working days"],
               purpose="negative control: a good agent reports no legal conflicts")))

# 2. Non-compete + training bond
DOCS.append(dict(
    id="doc02", file="02_noncompete_bond.docx", fmt="docx",
    text=make("Employment Agreement",
              "Between Brightwave Technologies Pvt. Ltd. (Company) and Arjun Mehta (Employee), Backend Developer, Chennai.",
              [("Notice Period", "Either party may terminate by giving 90 days' written notice."),
               ("Training Bond", "The Employee must serve at least 24 months. If the Employee resigns earlier, the Employee shall pay Rs. 4,00,000 as liquidated damages for training costs."),
               ("Non-Compete", "For 2 years after leaving, the Employee shall not work for or advise any competitor of the Company anywhere in India."),
               ("Intellectual Property", "All work created during employment belongs to the Company.")]),
    truth=dict(issues=[
        dict(clause=3, topic="post-employment non-compete", expected_act="Indian Contract Act, 1872", expected_section="27", confidence="high"),
        dict(clause=2, topic="training bond / liquidated damages", expected_act="Indian Contract Act, 1872", expected_section="74", confidence="medium")],
        should_not_flag=["90-day notice period (no fixed statutory maximum for most IT roles; depends on contract)", "IP assignment"],
        purpose="tests query rewriting (non-compete -> restraint of trade) and bond handling")))

# 3. Slow final settlement + broad deductions
DOCS.append(dict(
    id="doc03", file="03_slow_settlement.pdf", fmt="pdf",
    text=make("Offer Letter and Terms of Employment",
              "Issued by Cobalt Systems Pvt. Ltd. to Sneha Iyer for the role of QA Engineer, Hyderabad.",
              [("Salary", "Gross salary is Rs. 9,00,000 per year, paid monthly."),
               ("Full and Final Settlement", "On resignation, all dues will be settled within 60 days after the last working day, after clearance of all company assets."),
               ("Deductions", "The Company may deduct any amount from salary or final dues for losses, damages, policy violations or any other reason, without limit."),
               ("Notice Period", "Notice period is 60 days.")]),
    truth=dict(issues=[
        dict(clause=2, topic="final settlement timing", expected_act="Code on Wages, 2019", expected_section="17", confidence="medium"),
        dict(clause=3, topic="unlimited deductions from wages", expected_act="Code on Wages, 2019", expected_section="18", confidence="medium")],
        should_not_flag=["60-day notice period"],
        purpose="the key finding: 60 days vs the two-working-days rule; also deductions")))

# 4. Fixed-term contract denying gratuity
DOCS.append(dict(
    id="doc04", file="04_fixed_term_gratuity.docx", fmt="docx",
    text=make("Fixed-Term Employment Contract",
              "Between Helix Digital Pvt. Ltd. and Karthik Rao, Data Analyst, Pune. Term: 2 years from the date of joining.",
              [("Term", "This is a fixed-term contract for 2 years and will end automatically on expiry."),
               ("Gratuity", "As a fixed-term employee, the Employee is not eligible for gratuity under any circumstances."),
               ("Provident Fund", "The Employee is not covered by provident fund during the fixed term."),
               ("Renewal", "Renewal is at the sole discretion of the Company.")]),
    truth=dict(issues=[
        dict(clause=2, topic="gratuity for fixed-term employees", expected_act="Code on Social Security Act, 2020", expected_section="53", confidence="medium"),
        dict(clause=3, topic="provident fund coverage", expected_act="Code on Social Security Act, 2020", expected_section="verify", confidence="low")],
        should_not_flag=["automatic end of fixed term"],
        purpose="tests exception handling: the retrieved text carries conditions, answer must not say a flat 'no'")))

# 5. Leave policy with short maternity leave
DOCS.append(dict(
    id="doc05", file="05_leave_policy.txt", fmt="txt",
    text=make("Leave Policy",
              "Policy of Orbit Infotech Pvt. Ltd., applicable to all employees.",
              [("Casual Leave", "12 days per year."),
               ("Maternity Leave", "Female employees are entitled to 12 weeks of maternity leave for any number of children."),
               ("Sick Leave", "10 days per year with a medical certificate.")]),
    truth=dict(issues=[
        dict(clause=2, topic="maternity leave duration", expected_act="Code on Social Security Act, 2020", expected_section="verify (Chapter VI Maternity Benefit)", confidence="medium")],
        should_not_flag=["casual leave", "sick leave"],
        purpose="duration check against the Code on Social Security maternity chapter")))

# 6. POSH policy gaps
DOCS.append(dict(
    id="doc06", file="06_posh_policy.pdf", fmt="pdf",
    text=make("Workplace Conduct Policy",
              "Policy of Vertex Cloud Pvt. Ltd., which has about 150 employees across two offices.",
              [("Harassment", "The Company does not tolerate sexual harassment. Complaints may be sent to HR."),
               ("Complaint Window", "Complaints must be made within 7 days of the incident or they will not be considered."),
               ("Committee", "The Company has not constituted a separate committee because HR handles all complaints.")]),
    truth=dict(issues=[
        dict(clause=3, topic="Internal Committee requirement", expected_act="Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013", expected_section="4", confidence="high"),
        dict(clause=2, topic="time limit for complaints", expected_act="Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013", expected_section="9", confidence="medium")],
        should_not_flag=["zero tolerance statement"],
        purpose="tests POSH retrieval for a clean, well-defined law")))

# 7. Ambiguous termination letter: agent should hedge, not give a confident verdict
DOCS.append(dict(
    id="doc07", file="07_termination_letter.txt", fmt="txt",
    text=make("Termination Letter",
              "From Zenith Apps Pvt. Ltd. to Rahul Verma, Senior Developer, dated 4 March 2026.",
              [("Termination", "Your employment is terminated with immediate effect for performance reasons. No notice period or pay in lieu will be provided."),
               ("Dues", "Your pending salary will be paid in the next payroll cycle.")]),
    truth=dict(issues=[],
        should_not_flag=[],
        expected_behavior="hedge: whether the employee is a 'worker', what the contract says, and state rules all matter. A confident verdict is a failure; suggesting a labour lawyer is correct.",
        purpose="tests refusal / uncertainty handling")))

# ---------- write the files ----------
for d in DOCS:
    p = OUT / d["file"]
    if d["fmt"] == "txt":
        p.write_text(d["text"], encoding="utf-8")
    elif d["fmt"] == "docx":
        from docx import Document
        doc = Document()
        for line in d["text"].split("\n"):
            if line.strip():
                doc.add_paragraph(line)
        doc.save(str(p))
    elif d["fmt"] == "pdf":
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        styles = getSampleStyleSheet()
        story = []
        for line in d["text"].split("\n"):
            story.append(Paragraph(line.replace("&", "&amp;"), styles["Normal"]) if line.strip() else Spacer(1, 8))
        SimpleDocTemplate(str(p), pagesize=A4).build(story)

key = [dict(id=d["id"], file=d["file"], **d["truth"]) for d in DOCS]
(OUT / "answer_key.json").write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")

# ---------- question test set ----------
Q = [
    # (question, document, expected_route, expected_act, expected_section, should_refuse)
    ("Can my company enforce a non-compete after I resign?", "02_noncompete_bond.docx", "doc_question", "Indian Contract Act, 1872", "27", False),
    ("Is the Rs 4 lakh bond in my agreement valid?", "02_noncompete_bond.docx", "doc_question", "Indian Contract Act, 1872", "74", False),
    ("How soon must my final settlement be paid after I resign?", "03_slow_settlement.pdf", "doc_question", "Code on Wages, 2019", "17", False),
    ("Can the company deduct any amount from my salary?", "03_slow_settlement.pdf", "doc_question", "Code on Wages, 2019", "18", False),
    ("I am on a fixed-term contract, am I eligible for gratuity?", "04_fixed_term_gratuity.docx", "doc_question", "Code on Social Security Act, 2020", "53", False),
    ("Is 12 weeks of maternity leave enough under the law?", "05_leave_policy.txt", "doc_question", "Code on Social Security Act, 2020", "verify", False),
    ("Does our company need an Internal Committee for sexual harassment?", "06_posh_policy.pdf", "doc_question", "Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013", "4", False),
    ("Is the 7-day complaint window in our policy valid?", "06_posh_policy.pdf", "doc_question", "Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013", "9", False),
    ("Was my termination legal?", "07_termination_letter.txt", "doc_question", "", "", True),
    ("Review this contract and tell me what to be careful about", "02_noncompete_bond.docx", "review", "", "", False),
    ("Review this agreement", "01_clean_agreement.txt", "review", "", "", False),
    ("Am I eligible for gratuity after 4 years?", None, "law_question", "Code on Social Security Act, 2020", "53", False),
    ("What is the maternity leave duration?", None, "law_question", "Code on Social Security Act, 2020", "verify", False),
    ("How long does the company have to pay my dues after I leave?", None, "law_question", "Code on Wages, 2019", "17", False),
    ("Can I be forced to work overtime without extra pay?", None, "law_question", "Code on Wages, 2019", "verify", False),
    ("What is the best biryani in Chennai?", None, "out_of_scope", "", "", True),
    ("Should I sue my employer?", None, "law_question", "", "", True),
    ("Write me a Python script to sort a list", None, "out_of_scope", "", "", True),
]
with open(OUT / "qa_testset.jsonl", "w", encoding="utf-8") as f:
    for i, (q, doc, route, act, sec, refuse) in enumerate(Q, 1):
        f.write(json.dumps(dict(id=f"q{i:02d}", question=q, document=doc, expected_route=route,
                                expected_act=act, expected_section=sec, should_refuse=refuse),
                           ensure_ascii=False) + "\n")

print(f"Wrote {len(DOCS)} documents, answer_key.json and {len(Q)} questions to {OUT}")
