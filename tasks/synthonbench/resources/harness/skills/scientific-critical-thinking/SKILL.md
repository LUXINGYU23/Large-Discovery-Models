---
name: scientific-critical-thinking
description: Interpret chemical measurements and literature without confusing hypotheses, selection effects, or correlated candidates with independent evidence. Use when deciding whether to concentrate, explore, or revise a mechanism.
license: MIT
metadata:
  source: K-Dense-AI/scientific-agent-skills
  source-commit: 1e5eeffbdad3749125afe7ab48a39694e27f181c
---

# Evaluate Evidence During Chemical Optimization

Retrieve relevant measured candidates and their original research annotations.
Check the actual changes and competing explanations before attributing an
effect. A proxy, fitted prediction, proposal vote, or selection decision is
not a new objective measurement. Several annotations for the same candidate
are not independent evidence.

Repeated high scores in one family support local refinement, not global
optimality. A weak condition or molecule may reflect its particular background,
rather than refute the whole mechanism. Maintain credible unresolved
alternatives with measured IDs and the next comparison that would change your
allocation. Different IDs alone do not establish scientific diversity.

Search primary experiments to answer a concrete question or challenge a
hypothesis. Inspect chemistry, substrate or assay context, controls, and the
measured endpoint. Record the source URL or DOI and the specific supported
claim. A snippet is only a lead, and inaccessible text is not negative evidence.
Use existing web tools within their budgets and Context7 for library APIs;
this Skill requires no additional search service.

The measured panel is adaptively selected, not a random sample of the domain.
A rising batch average can reflect denser sampling of one good family.
Unselected candidates have no negative label and remain eligible until
actually evaluated. Missing calibration is uncertainty, not evidence of harm.

Keep concise notes covering the leading explanation, strongest viable rival,
supporting and contradicting measured IDs, and a useful next test. Update
beliefs after feedback without rewriting original submission rationales.
Do not turn every round into a review report or search only to justify a winner.
