# Impact statement

Rakshaka AI protects Sri Lankans who are new to digital money from scam messages in their own language,
and keeps protecting them when scammers switch tactics.

**Who is protected.** First-time mobile-banking users, pensioners and welfare recipients who receive scam SMS
and WhatsApp messages in Sinhala, Tamil or Singlish. The Central Bank says scams hit hardest among
"rural communities, low-income households, senior citizens" and young people ([CBSL Governor, 2 June 2026][bis]).
Two of those groups are large and well defined:

| Group | Size | Scams they see |
|---|---|---|
| Government pensioners | 727,092 (Feb 2025) ([Department of Pensions][pen]) | Fake "pension re-verification" calls and messages asking for a PIN |
| Aswesuma welfare beneficiary families | 1,251,522 (Aug 2026) ([Ada Derana][asw]) | Fake "payment pending, update your bank details" links |

**Reach.** Sri Lanka has 13.9 million internet users and 30.3 million mobile connections, about 130% of the
population ([DataReportal, late 2025][dr]). Every one of them can receive these messages. Our pilot target is
**10,000 users in six months** through one partner channel (see below), then the roughly 2 million pensioner and
Aswesuma households above. These are our targets, not results.

**Harm prevented.** Sri Lanka CERT received more than 12,650 complaints about social media misuse and cyber
incidents in 2025, including financial scams ([Sunday Times][st]). It logged more than 360 online financial fraud
incidents in a single month of 2026 ([Ada Derana][ad]). We found no official national total for money lost, so we
do not claim a rupee figure. A pilot would measure harm prevented directly:

- scam messages flagged before the user acted on them (on our held-out test set, 0 of 190 scams were missed);
- time to detect a new tactic: the drift alert fires within one 50-message window of the shift;
- the false-alarm rate, the cost to users (11 of 74 legitimate test messages were flagged);
- reports passed on to SLCERT about new scam tactics.

**Path to adoption.**

1. **Pilot (months 0–6):** a "forward a message to check it" WhatsApp number, plus the web checker, run with one bank's
   fraud helpline. The bank's fraud team acts as the human reviewers, so the review-and-retrain loop runs on real reports.
2. **Expand (months 6–12):** add pensioner and Aswesuma help desks at Divisional Secretariats. Align the messaging with
   CBSL's "Be Scam Proof" campaign, and share drift alerts about new tactics with SLCERT.
3. **Scale:** offer the API to banks and telcos for in-app and SMS-gateway checks.

Before going live: replace synthetic training data with consented, reviewed real reports; meet the Personal Data
Protection Act No. 9 of 2022 (consent, retention limits); and raise Sinhala accuracy (0.63 balanced accuracy on
test today).

**Why it keeps working.** Drift monitoring, a safe "uncertain" fallback, human review and one-click retraining
mean new scam tactics are caught as they appear, instead of the model failing silently.

[bis]: https://www.bis.org/review/r260602j.htm
[pen]: https://pensions.gov.lk/images/statistics/pdf/FEB-2025.pdf
[asw]: https://adaderana.lk/news/cmtwp3mlo0000356px4jrnmfw
[dr]: https://datareportal.com/reports/digital-2026-sri-lanka
[st]: https://sundaytimes.lk/online/news-online/CERT-records-over-12-000-cybersecurity-complaints-in-2025/2-1151060
[ad]: https://adaderana.lk/news/cmpxdqu4o0005356p3zevprhe
