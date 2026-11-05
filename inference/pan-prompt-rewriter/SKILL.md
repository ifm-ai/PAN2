---
name: pan-prompt-rewriter
description: Rewrite text-to-video or image-to-video requests with the unchanged positive policy and fixed no-actor-ID negative. Preserve metadata and durations, inspect supplied first frames, and validate serialized prompts and production token budgets. Not for video generation.
---

# PAN2 Prompt Rewriter

Use the complete [system_prompt.md](system_prompt.md) for positive authoring. Apply it directly with the current authoring model; this skill does not select a model, call a rewrite API, or generate videos.

## Authoring and assembly are separate stages

The system prompt's instruction to return only `refined_prompt` governs the **intermediate positive-authoring response**. This skill also performs the caller's assembly stage: its final output contains **both** `refined_prompt` and `negative_prompt` as JSON strings, together with preserved source metadata. Do not change the system prompt to reconcile those two stages.

1. Accept a plain-language request, one JSON record, or an ordered list. Wrap plain text in `original_prompt`. Preserve all metadata values and record order. When rewriting existing records, save prior captions separately and exclude only `refined_prompt` and `negative_prompt` from the clean request. Do not use old captions as examples to copy.
2. Resolve `duration_sec` or `target_duration_sec` exactly as the policy specifies; if neither is present, add `duration_sec: 13`. Preserve supplied durations. Do not silently round invalid durations or reconcile conflicting aliases.
3. For I2V, actually view the supplied first-frame image before authoring. Preserve its visible state and crop; requested additions belong in continuous later events as the policy specifies. A filename alone is not visual evidence. If the image is inaccessible, ask for the image or permission to use text-only conditioning. For T2V, no image is required. Before drafting I2V, note visible held objects and occupied hands; check the first action accounts for these so an existing object is not silently dropped or replaced.
4. Read the full positive policy and author each request from its original text and, when provided, viewed image. Produce the intermediate object `{"refined_prompt":"<serialized positive JSON>"}`. Use `json.dumps(positive, ensure_ascii=False, separators=(",", ":"), allow_nan=False)`; do not double-encode. Keep the author's choices governed by the policy rather than imposing a new event count, camera plan, pacing template, or aesthetic. Before drafting, apply the policy's distinction between independently acting subjects and genuinely collective behavior; distinct actions and held-object roles need distinct performer attribution. Prefer a suitable object already established by the request or image over inventing a replacement and an extra introduction action; retain any explicitly requested additional object. Treat the token limit as a ceiling, not an instruction to minimize caption length: retain useful scene-specific spatial, material and action detail without padding or repeating it.
5. Assemble with [scripts/prompt_tools.py](scripts/prompt_tools.py). It preserves metadata and the authored positive, validates structure, and attaches [assets/generic_negative.txt](assets/generic_negative.txt). This is exactly the no-actor-ID template. Never author, paraphrase, scene-specialize, add modules to, or shorten the negative. Its subject declaration remains, but its contained action has no `actor_id`. Its negative camera tags and artifact list intentionally differ from the positive schema.
6. Review the result against the source and the viewed image for requested actions, outcomes, initial state, continuity, and camera behavior. Run the helper's production-token validation when the deployed tokenizer is available. Preserve each emitted positive response unchanged. If validation or comparison finds a problem, record it separately; do not patch the returned caption. For a skill-development iteration, revise these workflow instructions only where the observed issue warrants it, rebuild from the handoff, and generate a fresh response from the original request and image. Keep the positive policy unchanged unless the user explicitly requests its revision. Save every attempt separately, and do not retry indefinitely. Never fix a negative overflow by editing the fixed template. Report unresolved failures.

The negative changes only its three `time_range_sec` values from `0 to {duration_sec}` to `0 to D`, using plain decimal notation. All other characters in the stripped template remain identical. Both prompt strings decode once into their seven-key objects. The final record is the clean source record plus these two strings; a batch is an equally sized list in source order. Validation receipts belong in separate files or brief accompanying status, never inside the captions.

## Run the helpers

Save `request.json` as clean source metadata and `positive.json` as the intermediate positive-only response. For a batch, both are lists in matching order. Replace `SKILL_DIR` with this skill's directory and choose new output paths:

```bash
python3 -B SKILL_DIR/scripts/prompt_tools.py assemble \
  --request request.json --positive positive.json --output candidate.json
python3 -B SKILL_DIR/scripts/prompt_tools.py validate \
  --request request.json --positive positive.json --candidate candidate.json \
  --report validation.json --tokenizer /path/to/deployed/text_encoder/vlm
```

The validator uses the same chat-template crop and suffix count as the API rewriter. It requires a Python environment with `transformers` and the deployed tokenizer's dependencies and local assets. Positive soft limit: 1,480; hard capacity: 1,536 **per caption**. It does not append negative modules. Do not substitute a generic LLM tokenizer.

Without those assets, omit `--tokenizer` for a structure/preservation check. This intentionally exits 2 with `tokenizer_ok: false`; label the result **Unverified draft — production encoder-token counts have not been verified.** Exit 0 indicates complete mechanical validation; exit 1 indicates a checked failure. A receipt is bound to the exact candidate; any edit requires another check. Keep failed drafts and receipts when correcting.

Without code, apply the same schema and exact template copy with only the three duration substitutions, and label the result unverified. Do not invent counts or receipts. Similarity to prior results means fidelity to the same request, image, authoring policy, output schema and fixed negative; positive wording and scene choices can vary with the model and sampling. Prompt checks do not establish generated-video quality.

## Compare skill revisions

Keep the authoring model, reasoning configuration, input records and images consistent across trials. Use saved API captions only for evaluation, not as per-record examples or text to copy into rewrites. Preserve raw generated positives and assemble them deterministically; never present a manually corrected output as skill-generated. Record the instruction snapshot, test provenance, failures and comparison separately. Distinguish source fidelity and image grounding from exact wording, discretionary staging and mechanical validity. If the same session has already seen references, disclose that the trial is not blind.
