"""Curated knowledge for the OKF bundle: what each Act covers, and common employment-law
topics mapped to the statute sections that govern them.

This is the "LLM wiki" layer OKF is designed for: it bridges everyday wording
("non-compete", "F&F settlement") to statutory wording ("agreement in restraint of trade").
The baseline RAG agent receives the same vocabulary hints in its query-rewriting prompt,
so neither pipeline gets knowledge the other lacks.
"""

CW = "Code on Wages, 2019"
SS = "Code on Social Security Act, 2020"
ICA = "Indian Contract Act, 1872"
POSH = "Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013"

ACT_INFO = {
    CW: "Wages: minimum wages, payment of wages and time limits, deductions and fines, overtime, "
        "equal pay regardless of gender, bonus, claims for unpaid dues.",
    SS: "Social security: provident fund, employees' state insurance, gratuity, maternity benefit, "
        "employee compensation for injury, gig and unorganised workers.",
    ICA: "General contract law: when agreements are valid or void, free consent, restraint of trade "
         "(non-compete), compensation and penalties for breach of contract.",
    POSH: "Prevention of sexual harassment of women at the workplace: Internal Committee, complaints, "
          "inquiry, employer duties and penalties.",
    "Payment of Gratuity Act, 1972": "Gratuity on leaving after continuous service (older law, now replaced).",
    "Maternity Benefit Act, 1961": "Maternity leave and benefits for women employees (older law, now replaced).",
    "Employees Provident Funds and Miscellaneous Provisions Act, 1952": "Provident fund and pension schemes (older law, now replaced).",
    "Payment of Wages Act, 1936": "Timely payment of wages and permitted deductions (older law, now replaced).",
    "Minimum Wages Act, 1948": "Fixing and paying minimum wages (older law, now replaced).",
    "Industrial Disputes Act, 1947": "Industrial disputes, conciliation, strikes and lockouts (older law, now replaced).",
    "Industrial Employment (Standing Orders) Act, 1946": "Standing orders on conditions of employment (older law, now replaced).",
}

# slug, title, description, aliases, [(act, section), ...], note
TOPICS = [
    ("non-compete", "Non-compete and restraint after leaving a job",
     "Clauses stopping an employee from joining a competitor or starting a business after leaving.",
     ["non-compete", "non compete", "restrictive covenant", "non-solicitation", "restraint of trade",
      "join a competitor", "after resignation", "post-employment restriction"],
     [(ICA, "27"), (ICA, "23"), (ICA, "24")],
     "Indian statutes describe these as agreements 'in restraint of trade'. Read Section 27 and its "
     "exception carefully; restrictions during employment and after employment can be treated differently."),
    ("service-bond", "Training bonds, service bonds and liquidated damages",
     "Clauses requiring an employee to pay a fixed sum if they leave before a minimum period.",
     ["training bond", "service bond", "employment bond", "liquidated damages", "penalty clause",
      "bond amount", "breach of contract", "leave before", "lock-in period"],
     [(ICA, "74"), (ICA, "73"), (ICA, "27")],
     "A sum named in a contract for breach is dealt with as compensation for breach where a penalty is "
     "stipulated (Section 74); general compensation for breach is in Section 73."),
    ("final-settlement", "Final settlement of dues after resignation or termination",
     "When wages and dues must be paid after an employee resigns, is removed, dismissed or retrenched.",
     ["full and final settlement", "F&F", "final settlement", "dues after resignation", "last salary",
      "relieving", "notice pay", "pending dues", "settlement delay"],
     [(CW, "17"), (CW, "43"), (CW, "45")],
     "The time limit for wages when employment ends is in the Code on Wages, Section 17."),
    ("wage-deductions", "Deductions and fines from salary",
     "What an employer may deduct from wages, limits on deductions, and fines.",
     ["salary deduction", "deduct from salary", "recovery from salary", "cut from wages", "fine",
      "deduction for damage", "advance recovery", "loan recovery"],
     [(CW, "18"), (CW, "19"), (CW, "20"), (CW, "21"), (CW, "23"), (CW, "24")],
     "Section 18 lists the deductions that may be made and the overall limit; Sections 19-24 cover specific kinds."),
    ("wage-payment", "Timely payment of salary and wage periods",
     "When and how salary must be paid, and how long a wage period may be.",
     ["salary delay", "late salary", "wage period", "pay date", "mode of payment", "salary not paid"],
     [(CW, "15"), (CW, "16"), (CW, "17")],
     ""),
    ("minimum-wages", "Minimum wages and floor wage",
     "The minimum rate of wages and how it is fixed.",
     ["minimum wage", "minimum salary", "floor wage", "underpaid"],
     [(CW, "5"), (CW, "6"), (CW, "7"), (CW, "9")],
     ""),
    ("overtime-hours", "Working hours and overtime pay",
     "Normal working day and wages for overtime work.",
     ["overtime", "extra hours", "working hours", "OT pay", "work beyond hours", "long hours"],
     [(CW, "14"), (CW, "13")],
     ""),
    ("equal-pay", "Equal pay and gender discrimination in wages",
     "No discrimination in wages or recruitment on the ground of gender.",
     ["equal pay", "gender pay gap", "paid less than men", "discrimination in salary"],
     [(CW, "3"), (CW, "4")],
     ""),
    ("bonus", "Statutory bonus",
     "Eligibility for bonus, disqualification and time limit for paying bonus.",
     ["bonus", "annual bonus", "statutory bonus"],
     [(CW, "26"), (CW, "29"), (CW, "39")],
     ""),
    ("gratuity", "Gratuity",
     "Gratuity on leaving after continuous service, including fixed-term employment, and how it is calculated.",
     ["gratuity", "five years service", "5 years", "continuous service", "fixed term", "fixed-term contract",
      "retirement benefit"],
     [(SS, "53"), (SS, "54"), (SS, "56"), (SS, "55")],
     "Section 53 sets when gratuity is payable and contains special rules for fixed-term employment."),
    ("maternity-benefit", "Maternity benefit and leave",
     "Paid maternity leave, notice, miscarriage leave, nursing breaks, creche, and protection from dismissal.",
     ["maternity leave", "maternity benefit", "pregnancy", "pregnant", "26 weeks", "12 weeks", "creche",
      "nursing break", "miscarriage", "adoption"],
     [(SS, "60"), (SS, "59"), (SS, "61"), (SS, "62"), (SS, "65"), (SS, "66"), (SS, "67"), (SS, "68"), (SS, "69")],
     "The length of maternity benefit is in Section 60 of the Code on Social Security."),
    ("provident-fund", "Provident fund",
     "Provident fund schemes and contributions by employers and employees.",
     ["PF", "EPF", "provident fund", "PF contribution", "pension scheme"],
     [(SS, "15"), (SS, "16"), (SS, "17")],
     ""),
    ("sexual-harassment", "Sexual harassment at the workplace (POSH)",
     "Internal Committee, how and when to complain, inquiry, employer duties and penalties.",
     ["POSH", "sexual harassment", "internal committee", "ICC", "harassment complaint",
      "complaint time limit", "inappropriate behaviour"],
     [(POSH, "4"), (POSH, "9"), (POSH, "3"), (POSH, "11"), (POSH, "13"), (POSH, "19"), (POSH, "26")],
     "Constitution of the Internal Committee is in Section 4; the time limit for a complaint is in Section 9."),
    ("contract-validity", "Validity of employment agreements",
     "When an agreement is a valid contract and when terms are void.",
     ["valid contract", "void clause", "free consent", "unlawful", "signed under pressure", "unfair terms"],
     [(ICA, "10"), (ICA, "14"), (ICA, "23"), (ICA, "24"), (ICA, "29")],
     ""),
    ("termination", "Termination of employment",
     "Ending employment: dues on termination, protection during pregnancy, compensation for breach.",
     ["termination", "fired", "terminated", "dismissal", "layoff", "retrenchment", "notice period",
      "wrongful termination"],
     [(CW, "17"), (SS, "68"), (ICA, "73")],
     "Note: retrenchment and notice rules of the Industrial Relations Code, 2020 are not in this knowledge base."),
    ("waiver-of-rights", "Agreements that give up legal rights",
     "Whether an employee can sign away rights given by the wage laws.",
     ["waiver", "give up rights", "contracting out", "agreement against law", "sign away"],
     [(CW, "60"), (CW, "61")],
     ""),
]
