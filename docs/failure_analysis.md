# Failure Analysis: Recurring Failure Patterns & Root Causes

This document details the **top 5 recurring failure patterns** observed in the `@AppleSupport` customer care pipeline, with real representative examples and architectural root causes.

---

## 1. Failure Pattern 1: Noisy DM-Routing Precedents Poisoning Reply Grounding

### Hypothesis & Root Cause
A major flaw in historical Twitter customer service datasets is that human agents frequently used Twitter as a routing gateway rather than a resolution channel. When the retrieval system pulls historical resolution pairs based on the `went_quiet_after_brand` proxy, it often retrieves brand tweets that merely state: *"Please DM us your Apple ID and iOS version."* 
Because the RAG generation prompt instructs the LLM to ground its draft in retrieved historical precedents, the LLM often echoes the canned DM redirect even for simple informational queries (e.g., "Where do I toggle Night Shift?") that could have been resolved immediately in public.

### Real Examples
1. **Customer Tweet:** `"How can I turn on two-factor authentication on my Apple ID from my iPad settings?"`
   - *Retrieved Precedent:* `"We can certainly help with that. Please send us a DM with your Apple ID email so we can look into your account."`
   - *Drafted Reply:* `"To get this sorted out, please send us a DM with your account details."`
   - *Failure Impact:* Unnecessary human escalation and friction for a public, documented setting (`Settings > Apple ID > Password & Security > Turn On 2FA`).
2. **Customer Tweet:** `"My AirPods won't show up in Bluetooth on my MacBook Pro."`
   - *Retrieved Precedent:* `"We'd be glad to assist. DM us your macOS version and serial number."`
   - *Drafted Reply:* `"Reach out in DM with your serial number to troubleshoot."`
   - *Failure Impact:* Fails to provide the standard 15-second reset button troubleshooting.

---

## 2. Failure Pattern 2: Refund / Policy Collision on Technical Complaints

### Hypothesis & Root Cause
Customer queries often exhibit **compound intents**: a technical defect combined with an explicit financial demand (e.g., *"My iPhone battery died 2 hours after the update, I demand a refund for this piece of junk!"*).
Semantic similarity in embedding space (`nomic-embed-text`) heavily weights technical keywords like `battery`, `update`, `iOS`, matching technical troubleshooting precedents. However, the customer's actual operational need is financial dispute handling and churn prevention. If the classifier predicts `os_update_battery_drain` (Medium Risk) instead of `billing_subscription_refund` / `cancellation_churn_complaint` (High Risk), the escalation policy risks marking it as `AUTO_HANDLE`.

### Real Examples
1. **Customer Tweet:** `"Updated to iOS 11 and my battery is dead in 30 mins. I want my money back or a replacement unit immediately!"`
   - *Primary Classification:* `os_update_battery_drain` (Confidence: 0.72)
   - *Retrieved Precedent:* Historical battery troubleshooting advice (*"Check Settings > Battery to review app usage"*).
   - *Failure Impact:* Offering basic battery health checks to a customer demanding financial compensation triggers intense frustration and escalates churn risk.
2. **Customer Tweet:** `"I was charged $99 for an app subscription renewal that crashed my entire iPad. Refund me now!"`
   - *Primary Classification:* `app_crash_software_bug` or `billing_subscription_refund` ambiguity.

---

## 3. Failure Pattern 3: Passive-Aggressive Sarcasm Evading Frustration Rules

### Hypothesis & Root Cause
The deterministic escalation policy relies on explicit safety keywords (`lawyer`, `sue`, `refund`, `unacceptable`) and an ALL CAPS ratio threshold (`>= 55%` over 15+ characters).
Sarcastic or passive-aggressive customers frequently use grammatically polite words with lowercase letters to express extreme hostility. Standard zero/few-shot LLM classifiers without emotional sentiment fine-tuning often classify these as `other_general_inquiry` with moderate confidence, evading the escalation safety net.

### Real Examples
1. **Customer Tweet:** `"Thank you so much Apple for the incredible update that turned my $1,000 phone into a gorgeous paperweight. Truly stellar work."`
   - *Intent Classified:* `other_general_inquiry` / `os_update_battery_drain`
   - *ALL CAPS Ratio:* 0.0% (Zero uppercase words)
   - *Keyword Match:* 0 matches (contains `thank you`, `incredible`, `stellar`)
   - *Escalation Decision:* `AUTO_HANDLE` (Severe False Negative!)
   - *Drafted Reply:* `"Thanks for reaching out! We'd be happy to look into this with you..."`
2. **Customer Tweet:** `"Another day, another brilliant bug where my alarms don't go off. Lost my morning meeting, thanks a lot."`
   - *Escalation Decision:* `AUTO_HANDLE` when senior human intervention was required.

---

## 4. Failure Pattern 4: Pronoun & Anaphora Blindness in Multi-Turn Threads

### Hypothesis & Root Cause
In Twitter support threads, customers rarely restate their issue in subsequent turns. Instead, they reply with short follow-ups:
- *"It didn't work."*
- *"Tried that already, still happening."*
- *"Now the screen won't even turn on."*
Because our evaluation evaluates the customer message turn-by-turn without full recursive thread context concatenation, the retriever and classifier fail to resolve the pronoun `it`. The message is classified into `other_general_inquiry` with low confidence, or retrieves irrelevant precedents.

### Real Examples
1. **Turn 3 Customer Tweet:** `"Already restarted twice, it's still doing the exact same thing."`
   - *Standalone Retrieval:* Matches generic restart advice (circular loop!).
   - *Failure Impact:* Re-suggesting the exact step the customer just stated they performed.
2. **Turn 4 Customer Tweet:** `"I'm at the store now and they said they don't have parts."`
   - *Standalone Intent:* `other_general_inquiry`
   - *Root Issue:* Escalated hardware repair logistics.

---

## 5. Failure Pattern 5: Hallucinated Specifics on Hardware Repair Pricing

### Hypothesis & Root Cause
7B/8B local LLMs (`llama3.1:8b`) possess general world knowledge mixed with training data from prior years. When customers ask about cracked screens or battery replacements, the LLM occasionally attempts to generate specific pricing (e.g., *"Screen replacements for iPhone 7 are $129"*), which contradicts Apple's official regional service tier policies and creates liability.
Grounded generation prompt engineering mitigates this by flagging `insufficient_context: true`, but edge cases where precedents mention outdated pricing still bleed through into drafts.

### Real Examples
1. **Customer Tweet:** `"How much does it cost to fix a cracked iPhone X screen at Genius Bar?"`
   - *Precedent Retrieved:* General Apple Support link (`apple.co/repair`).
   - *Drafted Reply:* `"Screen repair for iPhone X starts at approximately $279, or $29 with AppleCare+..."`
   - *Failure Impact:* Risk of quoting incorrect, outdated, or region-incompatible pricing rather than directing the user to official diagnostic portals.
