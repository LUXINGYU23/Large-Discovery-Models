# DeepSeek Harness web tool source

Source: [deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness/tree/687f282e8625c6b8d11ff58211a90524be2e4317), revision `687f282e8625c6b8d11ff58211a90524be2e4317`. Copyright (c) 2026 DeepSeek, MIT (see LICENSE).

`contract.ts` preserves these upstream function bodies:

- `packages/web/tool-web/src/search.ts`: parseSearchArgs, sourceLabel, formatSearchOutput, runSearchQueries, mergeSearchResults.
- `packages/web/tool-web/src/fetch.ts`: parseFetchArgs, computeFetchOutput.
- `packages/web/web-deepseek-api/src/search.ts`: mapSearchResponse, boundedQuestion.
- `packages/web/web/src/index.ts`: capSources.

Constants and the external-content notice match upstream defaults. Changes at this boundary: local structural types replace framework imports; several private functions are exported; invalidResponse creates an ordinary Error; renderBody includes only the text branch, since the API already returns Markdown. No HTML backend, DOM converter, Cordis lifecycle, desktop card renderer, or account-sign-in service is included.

`../../deepseek-web.ts` adapts registration, schema validation, prompt guidance, tool result metadata, 30-second cancellation and API-key bootstrap to Pi. API field projection, query batching and output formatting follow upstream. As upstream does, it sends one request per unique query, no automatic retries; a failed query cancels and discards the batch. The tool call budget counts batches, each containing up to four API requests.

Project-specific differences are intentional: network domain filtering and forbidden benchmark queries remain enforced; optional original-question context is omitted when forbidden by policy; transport errors do not echo provider bodies or secrets. The latest user message supplies ori_question (whitespace collapsed and bounded to 128 Unicode characters). Pi's policy layer may append a separate tool-budget notice after the bounded fetch result.

Upstream's API provider supports a broader account/configuration framework. Importing the plugin would pull that framework into Pi; this source subset retains its relevant algorithms without a second agent runtime. When updating, compare these functions against a newly pinned upstream revision and run the behavioral tests on the remote test host.
