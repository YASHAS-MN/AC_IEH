# Phase 3 Scope — Testing Portal (IEH)

## Objective

Validate real-time behavioral authentication using the trained owner/impostor model.

This phase exists to test trust evolution and decision stability.

This phase does NOT improve training.

Training artifacts are frozen.

Primary goal:

Observe whether trust adapts correctly during real interaction.

---

# Inputs

## Model Artifacts

* checkpoints/best_model.pth
* checkpoints/scaler.pkl
* calibrated threshold
* evaluation/final_report.json

## Live Behavioral Inputs

Mouse:

* movement
* velocity
* trajectory
* curvature

Keyboard:

* dwell
* flight

Metadata:

* timestamp
* context
* session state

---

# Outputs

## Trust Output

Range:

0 → 100

Example:

90–100 → Fully trusted

70–89 → Verified

50–69 → Observe

30–49 → Challenge

0–29 → Recovery required

---

## Session Output

Portal shall emit:

```json
{
 "score": 0.07,
 "trust": 91,
 "state": "verified"
}
```

Displayed values:

* reconstruction score
* trust score
* state
* timeline

---

# States

## VERIFIED

User behavior strongly matches owner.

Action:

* normal operation

---

## OBSERVE

Small deviation detected.

Action:

* continue monitoring
* no restriction

---

## CHALLENGE

Moderate confidence drop.

Action:

* display warning
* request passive confirmation

---

## RESTRICT

Low confidence.

Action:

* disable protected actions
* maintain read access

---

## RECOVERY

Trust exhausted.

Action:

* stop protected access
* allow recovery flow only

---

# Transitions

VERIFIED
→ OBSERVE

Condition:

* trust < 80

---

OBSERVE
→ CHALLENGE

Condition:

* trust < 60

---

CHALLENGE
→ RESTRICT

Condition:

* trust < 40

---

RESTRICT
→ RECOVERY

Condition:

* trust < 20

---

Recovery path:

RECOVERY
→ VERIFIED

Condition:

* successful recovery

---

Trust recovery:

Trust may increase gradually after stable owner behavior.

Trust shall not instantly jump.

---

# Recovery

Recovery is exceptional.

Recovery is NOT normal login.

Recovery requires:

* user recovery secret
  OR
* secondary ownership proof

Recovery secret is not stored in plaintext.

Recovery resets trust.

Recovery shall generate audit logs.

---

# Out of Scope

Not included in Phase 3:

* retraining
* recollection
* gateway deployment
* browser enforcement
* kernel hooks
* anti-tamper
* encryption redesign
* production security
* cloud synchronization
* remote execution
* automatic retraining

---

# Success Criteria

Phase 3 is successful if:

✓ Portal remains stable

✓ Owner maintains trust

✓ Impostor trust decays

✓ Trust transitions are smooth

✓ Recovery works

✓ No retraining required

Failure conditions:

✗ trust oscillation

✗ permanent lockout

✗ immediate block decisions

✗ dependence on manual intervention

---

# Exit Condition

Phase 3 completes when:

Portal demonstrates:

Behavior
→ Trust
→ Decision
→ Recovery

without changing the trained model.
