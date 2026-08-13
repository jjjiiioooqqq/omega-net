# Recorded-trace contract check — pilot order form and terms

Status: **draft for counterparty and legal review; not executed; not legal advice**.

This draft cannot be offered or signed until every bracketed field is completed and the provider confirms that the governing law, tax, business-registration, insurance, payment, privacy, and commercial-email requirements are workable within the budget.

## 1. Parties and order

- Provider legal name: **[PROVIDER LEGAL NAME]**
- Provider form / jurisdiction / registration or tax identifier, if applicable: **[PROVIDER DETAILS]**
- Provider notice address: **[VALID PHYSICAL POSTAL ADDRESS]**
- Customer legal name and entity type: **[CUSTOMER]**
- Customer notice address: **[CUSTOMER ADDRESS]**
- Effective date: **[DATE]**
- Authorized customer technical owner: **[NAME / TITLE / EMAIL]**
- Fee: **US$600, paid through [APPROVED PROCESSOR OR FUNDED NEUTRAL ESCROW] only after conditional Input Acceptance and Customer approval of the frozen contract hash**
- Delivery target: five business days after Analysis Start under Section 3, using **[TIMEZONE]** business days.

## 2. Fixed service

Provider will perform one advisory structural check against a customer-approved recorded-trace contract for:

- one customer-controlled AI workflow;
- one baseline and one candidate system version;
- one to six scenarios;
- exactly five matched trials per scenario;
- five to 30 baseline/candidate pairs in total;
- no more than three distinct external tool names;
- JSON/JSONL inputs totaling no more than 5,000 raw event objects across both variants, including recorded tool and approval events; and
- one customer-supplied field map and one bounded conversion into Provider's provided, documented canonical schema.

The deliverables are: the versioned recorded-trace contract; normalized inputs and hashes; deterministic JSON and Markdown advisory verdicts; structural evidence by scenario; supplied-trial cost/latency summary; prioritized structural findings; and one 30-minute handoff. Customer, not Provider, makes every release, remediation, safety, legal, and production decision.

The service does not include log collection, a custom connector, free-form log parsing, CSV conversion, source-code review, model execution, semantic or factual grading, human-label calibration, authentication of approval events, proof that upstream runs were not omitted or copied, vulnerability research, penetration testing, production access, legal/compliance advice, certification, or remediation implementation.

## 3. Input acceptance and clock

The parties first execute this order and its data/confidentiality terms. Customer then provides the completed preflight/field map, written authorization, and complete synthetic/staging or already-recorded packet. Provider may ask one consolidated clarification. Provider will then either:

1. create the canonical contract, give Customer its SHA-256 hash, obtain Customer's written approval, and issue **Conditional Input Acceptance**; or
2. reject or propose a written re-scope before analysis begins.

After Conditional Input Acceptance, Customer prepays or funds the agreed escrow. **Analysis Start** occurs only when both the approved contract hash and cleared funds are in place. The five-business-day clock and non-refundable analysis phase start at Analysis Start. A later rule change requires a separately numbered contract version and Customer approval; the original result is retained and not silently rescored. Customer-caused delay in providing an agreed answer or scheduling the handoff tolls the delivery clock for the same elapsed period.

Provider will not accept production credentials, live-user access, personal data, protected health information, payment-card data, government identifiers, trade secrets unnecessary for the check, data belonging to another party, or data Customer is not authorized to disclose. Customer must replace real identities, domains, tokens, and secrets with synthetic values.

If Provider rejects the packet before Conditional Input Acceptance, no fee is collected. After Analysis Start, the fee is non-refundable except if Provider does not deliver the fixed deliverables and does not cure within five business days after written notice.

## 4. Acceptance; changes; failure to deliver

Delivery is accepted when Provider supplies all Section 2 deliverables in readable form and offers the handoff. Customer will report an objectively missing or unreadable deliverable within five business days; Provider will have five business days to cure. Disagreement with an advisory verdict, a finding of no structural defect, or a later production outcome is not non-delivery.

Any additional scenario, trial, external tool, event volume, format, clarification round, integration, semantic review, remediation, or live system access requires a signed change order. Provider may decline a change that would exceed the eight-hour fulfillment ceiling, the data boundary, competence, authorization, or risk limits.

## 5. Customer responsibilities and authorization

Customer represents that it:

- controls or is authorized by the controller of every supplied system and record;
- has authority to commission this check and bind Customer;
- has obtained any internal approvals needed to disclose the synthetic/staging records;
- will not supply prohibited data or credentials;
- is responsible for the accuracy and trusted capture of its suite, system identity, field map, run manifest, trace records, approval markers, source bindings, policies, labels, costs, and latencies; and
- will independently review the findings before any release or consequential action.

Customer acknowledges that a recorded `human_approval` marker is not proof that a human actually approved an action, and case/trial completeness relative to a declared suite is not proof that an upstream run was not omitted, copied, renumbered, or selectively excluded.

## 6. Data handling and security

Before payment, Provider will attach a completed data-processing schedule naming every storage/transfer location, processor or AI service, access method, backup limitation, and retention period. Provider will use Customer Trace Data only to perform the pilot, restrict working access to the minimum necessary, and not upload it to any AI service unless that service's contractual handling is verified in the schedule and Customer expressly approves it. Customer will transfer the packet through the approved method, never by placing credentials or regulated data in ordinary email. If the schedule cannot be completed honestly, Provider will reject the packet.

**Customer Trace Data** means supplied traces, field maps, policies, normalized customer records, and customer-specific technical inputs. If Provider rejects the packet or either party terminates before Analysis Start, Provider will delete Customer Trace Data within five business days after rejection or termination and provide a deletion record. For an accepted analysis, Provider will delete working and retained report copies containing Customer Trace Data within 30 calendar days after delivery and the offered handoff, whether or not Customer attends the handoff, and provide a deletion record. Normalized customer records remain Customer Data. Provider may retain the executed agreement, invoice/payment record, written authorization, non-reversible input/output hashes, and a deletion record for **[LEGAL/ACCOUNTING RETENTION PERIOD]**. Any backup or platform retention that prevents these deletion promises must be named in the schedule before transfer; otherwise that platform may not be used.

Provider will notify Customer without unreasonable delay after confirming unauthorized access to Customer Data in Provider's control and will reasonably cooperate on containment. Because this pilot excludes personal and regulated data, Customer must stop transfer and notify Provider if such data is discovered.

## 7. Confidentiality

Each party will protect the other's nonpublic information using at least reasonable care and use it only for this pilot. Confidential Information excludes information that the recipient can document was already known without duty, becomes public without breach, is received lawfully from another source without duty, or is independently developed without the other's Confidential Information. Legally compelled disclosure is permitted after notice when legally allowed.

No customer name, logo, result, quote, benchmark, or case study may be published without separate written permission. Provider will not place Customer Data, private implementation details, or customer-specific findings into generalized assets.

## 8. Intellectual property

Customer retains all rights in Customer Data, systems, policies, labels, normalized customer records, and Confidential Information. After full payment, Customer owns the text of its customer-specific narrative findings and may use, copy, modify, and share those findings for its internal business and with professional advisers under confidentiality.

Provider retains all pre-existing and independently developed code, evaluator logic, schemas, templates, report layout, test methods, canonicalization and approval-binding logic, generic field maps, generalized non-confidential know-how, and improvements that do not contain Customer Data or disclose Customer Confidential Information (**Background IP**). No source-code ownership transfer is included. Customer receives a perpetual, paid-up, non-exclusive license to use, reproduce, and share internally the portions of Background IP embedded in the delivered report as necessary to exercise its rights in the findings. Before signature, Provider must confirm chain of title and disclose any university, employer, contractor, platform, or third-party claim or license affecting Background IP.

## 9. Disclaimers and reliance boundary

THE SERVICE AND DELIVERABLES ARE ADVISORY AND PROVIDED ON AN “AS IS” BASIS TO THE MAXIMUM EXTENT PERMITTED BY LAW. PROVIDER DOES NOT WARRANT THAT THE TRACES ARE AUTHENTIC OR COMPLETE, THAT THE SYSTEM IS SECURE, SAFE, CORRECT, COMPLIANT, OR FREE OF DEFECTS, THAT EVERY FAILURE WILL BE FOUND, OR THAT AN UNSEEN OR PRODUCTION RUN WILL MATCH THE SUPPLIED RECORDS. THIS IS NOT A SECURITY AUDIT, PENETRATION TEST, CERTIFICATION, LEGAL OPINION, OR RELEASE AUTHORIZATION.

Customer must not represent the deliverable as a certification or outsource its release, legal, safety, or compliance judgment to Provider.

## 10. Liability allocation

To the maximum extent permitted by the governing law, neither party is liable for indirect, special, incidental, consequential, exemplary, or punitive damages, lost profits, lost revenue, lost data, or business interruption arising from this pilot. Provider's aggregate liability arising from the pilot will not exceed the fee actually paid, except to the extent a limit is prohibited by law or for Provider's fraud or willful misconduct. Customer remains responsible for every production deployment and external action.

The parties acknowledge that this allocation is a material basis of the fixed $600 price. It must be reviewed for enforceability in the chosen jurisdiction and any non-waivable liability must be identified before signature.

## 11. Term; termination; general

Either party may terminate before Analysis Start; no fee is collected before Conditional Input Acceptance and an unstarted funded escrow will be released under the processor's agreed instructions. After Analysis Start, Customer may stop the work, but the fee remains due unless Section 4's uncured non-delivery rule applies. Sections 5–11 survive as needed to give them effect.

Neither party may assign this order without written consent, except to a successor in a merger or sale of substantially all relevant assets that assumes the obligations. Neither party is the other's employee, agent, fiduciary, partner, or joint venturer. This order and an attached accepted scope/field map are the entire pilot agreement and may be changed only in a signed writing. Electronic signatures and counterparts are permitted.

- Governing law and exclusive forum: **[STATE/COUNTRY AND COURTS — MUST BE COMPLETED]**
- Required privacy/security addendum, if any: **[NONE / ATTACHED]**
- Required tax forms: **[COMPLETE]**
- Professional/cyber insurance status and disclosed limitations: **[COMPLETE]**

## 12. Signatures

**Provider**  
Name / title: [ ]  
Signature / date: [ ]

**Customer**  
Name / title: [ ]  
Signature / date: [ ]
