# Invalid JSON handling policy

LLMs occasionally emit malformed JSON even when the provider advertises
structured output (observed in practice as escaped closing quotes, raw
newlines inside strings, markdown fences, and token-limit truncation).
`geniie-lab` handles every such failure with one uniform, fully logged
pipeline (issue #13), identical across providers:

1. **Deterministic rule-based repair** (`geniie_lab/services/llm/json_repair.py`).
   A fixed, ordered list of pure text-transformation rules is applied
   cumulatively, validating after each step: strip code fences, extract the
   JSON object from surrounding prose, escape raw newlines inside strings,
   fix a trailing escaped quote, close unterminated strings/braces. The same
   raw output always yields the same repaired output. No model round-trip is
   involved. Applied rules are reported on stderr
   (`[json-repair] repaired <Model> with rules [...]`), so every rescue is
   visible in the run log and quantifiable afterwards.

2. **Bounded validation-feedback retry** (max 3 attempts per call). If the
   output stays invalid after repair — typically *semantic* failures such as
   enum violations or missing fields, which no text rule can fix — the exact
   validation error is appended to the outgoing conversation and the call is
   retried. The failed exchange is kept out of the session memory, so the
   stored conversation stays identical to a run that succeeded first try.
   Attempts are reported on stderr and every attempt's tokens are included
   in the record's `total_token`, keeping cost accounting truthful. This
   mirrors how large Bedrock experiments were run successfully.

3. **Skip the topic**. After the attempts are exhausted the service raises,
   the experiment runner records a `[WARNING] ... unrecoverable LLM output`
   for that topic, stops that topic's pipeline, and continues with the next
   topic. The experiment as a whole is never lost to a single bad output,
   and the skip is explicit in the log.

## How the schema reaches the model

The response schema, including every field `title` and `description` written
in `geniie-lab/response.py`, is sent as `response_format` with `strict: True`,
so the provider's grammar prevents a non-conforming response rather than
leaving it to the repair pipeline above.

Two settings change that, and they are independent.

- `schema_in_prompt` (default `False`) additionally appends the serialised
  schema to the outgoing user message, putting the field descriptions in the
  conversation, where models attend to them more than to the same text sent
  out of band. The schema then goes over the wire twice. The suffix rides on
  the outgoing copy only — session memory keeps the plain instruction, so
  logged conversations stay identical across providers and settings. Set it
  per experiment on `ModelDescription`; the Gemini service has no equivalent
  and ignores it.
- `json_object_fallback` (default `False`) drops the `json_schema`
  `response_format` for deployments that mis-handle it, leaving no grammar.
  Such an entry must also set `schema_in_prompt`, or nothing tells the model
  what to produce. Amazon Bedrock is the only registry entry using it today,
  and it sets both. Note the observation was made with one model there
  (gpt-oss-120b), while the setting is per provider, so every model on that
  entry loses grammar enforcement whether or not it needs to.

`total_token` is what the provider reported, always — the schema's tokens are
never netted out of it. Comparing providers where only some send the schema in
the prompt therefore means subtracting at analysis time; the schema for a
stage's response model can be re-serialised from `geniie_lab/response.py` and
measured, bearing in mind that those models change over time, so an estimate
made later is against today's schema rather than the one that ran.

## Reproducibility rationale

Feedback retries are sometimes avoided on reproducibility grounds. We take
the position that reproducibility in this setting comes from *visibility*,
not from refusing recovery: LLM inference is already nondeterministic at the
serving layer (batching, MoE routing) even at `temperature=0`, and the
FAQ's long-standing advice — resubmit the same instruction until it gets
through — is itself a retry, performed manually and unlogged. The pipeline
above automates it with every repair, retry, and skip recorded.

## Interaction with reasoning models

The reasoning trace ("thinking") of reasoning models travels outside the
structured output channel and can neither corrupt the JSON nor be affected
by repair. Reasoning-model support (trace capture and length control) is a
separate feature; the two compose without interaction.
