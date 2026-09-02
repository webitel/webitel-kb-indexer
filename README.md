# webitel-kb-indexer

Re-indexing worker of the Webitel Knowledge Base.

Consumes the `kb.reindex` queue, turns an article version into chunks and
embeddings, swaps the published version of the article and reports the outcome.

Contract and schema belong to `webitel-kb`: the envelope, the topology, the
acknowledgement rules and the `index_state` codes are defined in its
`docs/events.md`, and every `kb.*` migration lives there. This service
implements that document and ships no migration of its own.

## Run

```sh
uv sync
cp .env.example .env
uv run kb-indexer config   # effective configuration, credentials masked
uv run kb-indexer run
```

## Checks

```sh
uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest
```
