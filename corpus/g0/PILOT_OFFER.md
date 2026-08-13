# Recorded-trace contract check — founding pilot

Status: internal approval packet; not published or sent.

## The offer

A fixed-scope structural check for one tool-using AI workflow. The buyer supplies staging, synthetic, or already-recorded traces. Against a buyer-approved trace contract, the check detects missing required fields or exact expected values; claimed source IDs without matching recorded read-call arguments; unexpected or missing tools; failure of a declared required-call subsequence; external actions lacking buyer-declared immutable effect arguments; external actions without exactly one prior recorded approval marker bound to the call ID, canonical arguments, and expiry; absolute cost/latency budget failures; missing or duplicate case/trial labels relative to the declared suite; and candidate pass-rate regression against the supplied baseline. It does not judge semantic truth or upstream run selection.

- Price: **$600 prepaid or placed in neutral escrow**.
- Delivery: **five business days after Analysis Start**, which occurs only after complete inputs and the frozen contract hash are accepted and payment or escrow has cleared.
- Founder-time ceiling: **eight hours**. If the scope cannot fit, the pilot is declined or re-scoped before payment.
- System boundary: one workflow, one baseline and one candidate version, one to six buyer-approved scenarios, exactly five matched trials per scenario (5–30 baseline/candidate pairs), up to three distinct external tool names, and structured JSON/JSONL exports totaling at most 5,000 raw event records across both variants. An event is one raw recorded event object, including tool-call and approval records; envelope fields are not separately counted. The buyer must supply a field map; one bounded conversion into the provided, documented evaluator schema is included. CSV, free-form logs, log collection, and a custom integration are out of scope.
- Deliverables: a versioned recorded-trace contract, deterministic JSON and Markdown advisory verdict, reproducible structural evidence per scenario, operating-metric summary, prioritized findings, and a 30-minute handoff. The buyer makes the release decision.
- Data boundary: synthetic, staging, or buyer-recorded data only. No production credentials, live-user interaction, personal data, destructive action, social engineering, or access to systems the buyer is not authorized to test.
- Claim boundary: this is an advisory structural check of buyer-supplied recorded traces, **not** authentication of the traces or approvals, semantic evaluation, proof of upstream sampling completeness, a penetration test, legal/compliance opinion, certification, release authorization, safety guarantee, or promise that unseen behavior is safe.

The sequence is fixed: execute the short-form terms; receive and screen the permitted packet; resolve one consolidated clarification; freeze the buyer-approved canonical contract and SHA-256 hash; issue conditional Input Acceptance; collect cleared payment or funded escrow; then start analysis and the five-business-day clock. If the packet cannot be accepted, no payment is collected.

## Required buyer inputs

1. Written confirmation that the buyer controls the workflow and is authorized to supply the traces.
2. A plain-language description of expected behavior and prohibited external actions.
3. The trusted tool catalog, including which tools are read/local/control/external.
4. The approval rule for each external action.
5. The source identifiers and the trusted read-call arguments they represent.
6. Matched baseline and candidate trace exports with the same case IDs, trial IDs, suite version, and system identity; plus an optional buyer-supplied upstream run manifest for mechanical ID/hash reconciliation only. The manifest's own completeness and honesty are not verified.
7. Buyer-approved synthetic test identities, domains, tokens, and canaries.
8. For each external-action tool, the immutable effect arguments that must be recorded (for example target, amount, and content/payload digest). Approval records are treated only as buyer-supplied markers unless the buyer also supplies trusted-capture evidence.

## Ownership and confidentiality principles

- The buyer retains all rights in its data, systems, policies, and confidential information.
- After full payment, the buyer owns its client-specific narrative findings and may use the report internally subject to the executed agreement's reliance, confidentiality, and embedded-Background-IP boundaries.
- Pre-existing evaluator code, generic schemas, test methods, templates, canonicalization logic, approval-binding logic, and generalized know-how remain provider Background IP. No buyer data or confidential implementation detail may enter generalized assets.
- No logo, name, result, or case study is published without separate written permission.
- A short-form services agreement must make the authorization, confidentiality, data deletion, ownership boundary, warranty disclaimer, and liability cap explicit before traces are transferred.

These principles are commercial terms to be reviewed for the relevant jurisdiction; they are not legal advice and are not a substitute for an executed agreement.

## Frozen first paid-signal gate

The offer has not produced a paid signal. The first gate passes only when one unaffiliated company:

1. accepts this fixed boundary;
2. prepays or escrows at least $600 before delivery; and
3. supplies a workflow that can be delivered in eight founder hours or less.

Before the first send, freeze 30 qualified organizations and their evidence. The experiment starts when the first approved message is sent. It permits one individualized message and one follow-up per organization, 21 elapsed days, 12 total founder-active hours for the entire acquisition attempt—including cohort review, writing and sending, replies, calls, contracting, and payment/escrow setup—and $0 paid acquisition. Every frozen organization remains in the denominator even if a message bounces; confirmed delivery is reported separately and no row is replaced or reclassified after outcomes begin. Payment or funded escrow must clear inside both ceilings. Calls, compliments, free trials, letters of intent without money, and bounty income do not pass. Zero paid pilots from the frozen 30 stops this direct offer; the threshold may not be weakened after observing results. If fewer than 24 messages are confirmed delivered, the offer result is labeled inconclusive but the selected channel still fails.

One payment is evidence only of that buyer's willingness to pay; it does not validate a market, repeatability, productization, or asymmetric wealth. Productization remains prohibited until at least two unrelated buyers pay for materially the same automated recorded-trace check, logged delivery minutes show the repeated core in at least 60% of work, and at least one buyer purchases a second run or signs and funds a binding recurring order. Before treating it as a wealth vehicle, measured support, retention, gross margin, founder replacement, distribution, and transferable IP must also survive review.

## First message

Subject: one trace-contract check for [COMPANY]'s [PUBLICLY DOCUMENTED ACTION]

Hi [FIRST NAME] — I saw that [COMPANY] publicly describes its agent [SPECIFIC ACTION FROM SOURCE]. That creates a narrow release risk: a model or trace regression can cross from a plausible answer into the wrong external action.

I built a deterministic recorded-trace contract checker for exact-argument approval markers, source-ID/read-call binding, declared trial labels, required call subsequences, and cost/latency budgets. I have not tested your system or found a vulnerability. I am opening one $600 fixed pilot: one staging/synthetic workflow and reproducible advisory evidence within five business days. Your team makes the release decision. No production credentials, personal data, penetration testing, or compliance claims.

If that boundary is useful, reply and I will send the sample and fixed-scope packet. If not, reply “no” and I will not follow up.

[Commercial offer. To opt out of future email from me, reply “no.”]

[SENDER NAME]
[ROLE / BUSINESS NAME, IF ANY]
[VALID POSTAL ADDRESS REQUIRED BEFORE SEND]

## One permitted follow-up

Subject: Re: one trace-contract check for [COMPANY]'s [ACTION]

Hi [FIRST NAME] — closing the loop on the fixed recorded-trace contract check below. It is deliberately limited to one synthetic/staging workflow and $600 prepaid; no production access, release authorization, or broad security claim. If it is not relevant, no reply is needed and I will close this out.

[Commercial offer. To opt out of future email from me, reply “no.”]

[SENDER NAME]
[VALID POSTAL ADDRESS REQUIRED BEFORE SEND]

## Send controls

- No message is sent until the user approves the exact recipients, copy, sender identity, sending account, and compliant postal address.
- Every message must use accurate sender/header information, identify itself as a commercial offer, provide a functioning reply-based opt-out, and display the sender's valid physical postal address. The final jurisdiction screen remains mandatory.
- Use only a public organization/founder route that explicitly invites relevant inquiries; never infer or enrich a private address.
- Send manually and individually; no open pixel, link tracking, attachment in the first message, address harvesting, or automated sequence.
- Suppress an organization immediately on opt-out, complaint, wrong-person notice, or hard bounce. Pause the entire batch on one complaint.
- One follow-up only, no earlier than five business days. No contact after opt-out.
- Keep the reply opt-out working and monitored for at least 30 days after each send; honor and record opt-outs immediately and no later than the legal deadline.
- Do not imply a customer, certification, vulnerability, benchmark, or security credential that does not exist.

## Honest sample boundary

The frozen internal evaluator passed 48 of 48 unit and hostile-regression tests, including dataset-specific baseline/candidate version binding. SHA-256: evaluator `4c3a0cb2ef70064aa79356b30a16ad9d733da600104539f69d182065a05a0a14`; test suite `77a76ed640ba6d4ab53e9dfc0f7362c5a438dd05350822a675cfbfa0a8c459df`. The external sample archive contains synthetic fixtures, deterministic machine- and human-readable PASS/BLOCK reports, a manifest, and the exact internal-audit boundary; it deliberately excludes evaluator source code. Archive SHA-256: `32013411d032a18a8e95171882fdb2b144f9e1f8319094adc6d17d1e6d271eb3`. It demonstrates fail-closed behavior only for the declared structural contract and tested threat model. It does not demonstrate semantic accuracy, approval authenticity, upstream sampling completeness, a production vulnerability, buyer demand, a moat, or professional security certification.
