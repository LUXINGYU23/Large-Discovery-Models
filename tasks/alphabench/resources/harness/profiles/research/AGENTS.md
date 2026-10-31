# AlphaBench T3 factor research

The current turn supplies the market, backend, objective, required candidate count and new evaluated search observations. Use `mcp__t3__get_contract` for the authoritative operator and field contract. Search history tools expose only evaluated search observations. Static validation is free; `mcp__t3__check_expression` is a paid backend check. A GP query tool is available only in the named LDM query variant and reports a frozen start-of-round posterior, not an Oracle measurement.

Write `/workspace/candidates.json` as a UTF-8 object containing only a `candidates` array. Each element has a short `name`, an `expression`, and optionally a `research_note`. Submit `candidates.json` with `submit_candidates`. The array must contain exactly `candidate_count` distinct canonical expressions that have never been evaluated, including failed earlier evaluations. Repair any indexed rejection and resubmit the complete array.

Research only the supplied search history and public operator contract. Do not seek validation/test labels, Host files, market archives or provider credentials.
