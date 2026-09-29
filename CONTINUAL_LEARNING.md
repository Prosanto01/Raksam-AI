# Raksam Continual Learning

Raksam now has an experience-replay layer in addition to public-web learning.
It records conversations, action plans, device state, results, successes and
failures with provenance. Raw experiences are quarantined; only trusted or
explicitly approved events become supervised replay data.

## Sources of learning

1. **Public knowledge** — approved web sources through the existing crawler.
2. **Conversations** — stored automatically, but not treated as truth by default.
3. **User corrections** — can be labeled accepted/rejected using the experience ID.
4. **Successful actions** — automatically receive a higher trust score when the
   permission gate allowed them and the adapter reports a success.
5. **Failed/blocked actions** — recorded for analysis but excluded from training
   until deliberately approved.
6. **Device experiences** — device ID, state, requested action and result are
   preserved for replay.

## Automatic cycle

```bash
python continual_learning.py --loop 3600
```

The cycle builds replay files, collects configured web knowledge, and trains a
candidate when enough trusted experiences exist. The existing checkpoint is
kept separately so a bad candidate can be rolled back.

## Feedback

When an interaction returns `experience_id`, an application can label it after
user feedback. Accepted examples are promoted to high-trust replay; rejected
examples are retained as negative evidence but are not trained as desired
behavior.

This is continual learning, not magical self-correction: the system deliberately
avoids training directly on every generated answer because that can amplify
errors. Evaluation, provenance, replay balance and checkpoint validation remain
part of the loop.
