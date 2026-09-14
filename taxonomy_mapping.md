# Derived Intent Taxonomy for @AppleSupport

This taxonomy is derived by clustering customer-initiating messages from Kaggle Twitter Customer Support using `nomic-embed-text` embeddings and K-Means, then consolidating semantic clusters into 8 human-readable operational intents.

## Intent Summary Table

| Intent Label | Name | Risk Tier | Rationale |
| :--- | :--- | :--- | :--- |
| `account_access_security` | Account Access & Apple ID Security | **HIGH** | High risk because account lockouts prevent customer usage and deal with sensitive credentials and identity security. |
| `billing_subscription_refund` | Billing, Subscriptions & Refund Requests | **HIGH** | High risk due to direct financial impact, payment disputes, and chargeback threats. |
| `cancellation_churn_complaint` | Severe Dissatisfaction, Churn & Legal Escalations | **HIGH** | High risk because churn threats and hostile sentiment require senior human intervention and brand damage control. |
| `os_update_battery_drain` | OS Update, Battery Life & Overheating | **MEDIUM** | Medium risk because widespread post-update issues affect user experience but rarely involve account security or financial loss. |
| `hardware_physical_damage` | Hardware Malfunction & Physical Repair | **MEDIUM** | Medium risk because physical repairs require diagnostic check-ins and repair appointments. |
| `connectivity_network_bluetooth` | Connectivity, Wi-Fi, Bluetooth & Cellular | **MEDIUM** | Medium risk as it disrupts device functionality, often resolvable via network reset or troubleshooting steps. |
| `how_to_feature_inquiry` | How-To Guides & Settings Guidance | **LOW** | Low risk standard informational queries that can be safely automated with knowledge base links and instructions. |
| `other_general_inquiry` | Other General Inquiries | **LOW** | Low risk catch-all intent for non-urgent messages. |

## Detailed Intent Definitions

### `account_access_security` (Account Access & Apple ID Security)
- **Risk Tier**: `high`
- **Description**: Apple ID locked, forgotten password, 2FA verification codes not received, account hacking or takeover fears.
- **Escalation Rationale**: High risk because account lockouts prevent customer usage and deal with sensitive credentials and identity security.
- **Key Terms**: apple id, password, locked, verification code, two factor, account, login, reset, security

### `billing_subscription_refund` (Billing, Subscriptions & Refund Requests)
- **Risk Tier**: `high`
- **Description**: Unexpected charges, subscription cancellations, refund disputes, App Store purchase issues, double billing.
- **Escalation Rationale**: High risk due to direct financial impact, payment disputes, and chargeback threats.
- **Key Terms**: charged, refund, subscription, money, billing, cancel subscription, receipt, bank, purchase

### `cancellation_churn_complaint` (Severe Dissatisfaction, Churn & Legal Escalations)
- **Risk Tier**: `high`
- **Description**: Customers threatening to cancel services, switch to competitors (e.g. Android/Samsung), legal or regulatory threats.
- **Escalation Rationale**: High risk because churn threats and hostile sentiment require senior human intervention and brand damage control.
- **Key Terms**: never again, cancel, switch to samsung, lawyer, attorney, unacceptable, disgusting, terrible, worst service

### `os_update_battery_drain` (OS Update, Battery Life & Overheating)
- **Risk Tier**: `medium`
- **Description**: Battery draining quickly, device heating up, issues immediately following an iOS/macOS update, performance throttling.
- **Escalation Rationale**: Medium risk because widespread post-update issues affect user experience but rarely involve account security or financial loss.
- **Key Terms**: battery, ios, update, drain, heat, overheating, slow, lag, dying fast

### `hardware_physical_damage` (Hardware Malfunction & Physical Repair)
- **Risk Tier**: `medium`
- **Description**: Cracked screen, broken buttons, charging port failure, physical hardware defects, Genius Bar repair booking.
- **Escalation Rationale**: Medium risk because physical repairs require diagnostic check-ins and repair appointments.
- **Key Terms**: screen, cracked, broken, repair, button, charging port, hardware, genius bar, camera

### `connectivity_network_bluetooth` (Connectivity, Wi-Fi, Bluetooth & Cellular)
- **Risk Tier**: `medium`
- **Description**: Wi-Fi disconnecting, Bluetooth pairing failure (AirPods/CarPlay), cellular carrier signal drop, no service.
- **Escalation Rationale**: Medium risk as it disrupts device functionality, often resolvable via network reset or troubleshooting steps.
- **Key Terms**: wifi, bluetooth, airpods, connection, disconnecting, no service, carrier, cellular, pair

### `how_to_feature_inquiry` (How-To Guides & Settings Guidance)
- **Risk Tier**: `low`
- **Description**: Questions on how to configure settings, transfer data, use native features (Photos, iCloud backup, gestures).
- **Escalation Rationale**: Low risk standard informational queries that can be safely automated with knowledge base links and instructions.
- **Key Terms**: how do i, how to, settings, feature, transfer, enable, disable, where can i find

### `other_general_inquiry` (Other General Inquiries)
- **Risk Tier**: `low`
- **Description**: General praise, miscellaneous feedback, or non-critical questions outside defined operational buckets.
- **Escalation Rationale**: Low risk catch-all intent for non-urgent messages.
- **Key Terms**: thanks, question, apple, release date, store hours, info

## Unsupervised Cluster Inspection Summary

#### Cluster 0 (size: 3)
- **Top TF-IDF Terms**: `space, lock, settings, know, rid, change`
- **Centroid Exemplar**: *"I don’t know. I need the space though. How do I get rid of some snapshots"*

#### Cluster 1 (size: 19)
- **Top TF-IDF Terms**: `just, thank, let, cupertino, yes, email`
- **Centroid Exemplar**: *"And the new ones tauntingly pretend they are saved- it’s just when I click on them or try to send them that the grey screen shows"*

#### Cluster 2 (size: 20)
- **Top TF-IDF Terms**: `update, just, iphone, 11, updated, yes`
- **Centroid Exemplar**: *"iPhone 6 and yes it’s got the latest software update"*

#### Cluster 3 (size: 19)
- **Top TF-IDF Terms**: `letter, type, annoying, change, happening, hey`
- **Centroid Exemplar**: *"tell me why I️’I’ve updated my phone twice and I️ can’t type letter I️ without getting a letter A and a question mark"*

#### Cluster 4 (size: 24)
- **Top TF-IDF Terms**: `11, ios, ios11, keyboard, fix, just`
- **Centroid Exemplar**: *"still happening, including on iOS 11.2 beta now. any ideas?"*

#### Cluster 5 (size: 20)
- **Top TF-IDF Terms**: `music, control, phone, watch, apps, playing`
- **Centroid Exemplar**: *"I can’t download songs. Progress icon keeps spinning. On both data + WiFi. Stream works though. Can’t add songs to playlists."*

#### Cluster 6 (size: 23)
- **Top TF-IDF Terms**: `battery, happening, life, thanks, closed, hello`
- **Centroid Exemplar**: *"It’s showing it’s connected and the battery is decreasing."*

#### Cluster 7 (size: 22)
- **Top TF-IDF Terms**: `iphone, phone, does, help, plus, new`
- **Centroid Exemplar**: *"HELP ME. I have a problem with my iPhone 7😑"*
