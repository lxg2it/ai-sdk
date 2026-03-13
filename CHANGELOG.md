# Changelog

All notable changes to `@lxg2it/ai` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-14

### Added

- `ModelRouter` client class with `chat()`, `stream()`, `run()`, and `embed()` methods
- Typed routing tiers (`economy` | `standard` | `premium`) as first-class TypeScript types
- Routing preferences (`balanced` | `cheap` | `fast` | `quality` | `coding`)
- `include_reasoning` option for chain-of-thought when available
- Streaming with typed events: `reasoning`, `content`, `tool_call_delta`, `tool_call`, `done`, `error`
- Routing metadata surface (`model`, `provider`, `tier`, `latency_ms`) extracted from `X-Model-Router-*` response headers
- `tool()` factory for Zod-based, type-safe tool definitions
- `run()` agentic tool execution loop with automatic dispatch and iteration safety
- `embed()` for text embeddings via Model Router
- Vercel AI SDK provider adapter (`@lxg2it/ai/vercel`) — `createModelRouter()` factory
- Full TypeScript types, strict mode, ESM-only output
