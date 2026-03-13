# @lxg2it/ai

TypeScript SDK for the [Model Router](https://api.lxg2it.com) — route by intent, not by model name.

```bash
npm install @lxg2it/ai openai
```

## Why

The Model Router routes LLM requests by **intent tier** (`economy`, `standard`, `premium`) rather than specific model names. When better models are released, routing updates automatically — your code doesn't change.

This SDK wraps the OpenAI SDK to expose those routing concepts as first-class TypeScript types, and adds:

- **Streaming with reasoning** — typed events for thinking content, text, and tool calls
- **Tool execution loops** — agentic `run()` with automatic tool dispatch
- **Routing metadata** — see which model and provider actually handled each request
- **Vercel AI SDK integration** — one-line swap for Next.js and React apps

## Quick start

```typescript
import { ModelRouter } from '@lxg2it/ai';

const ai = new ModelRouter({ apiKey: 'mr_sk_...' });

// Chat
const response = await ai.chat({
  tier: 'standard',
  messages: [{ role: 'user', content: 'Hello!' }],
});

console.log(response.content);
console.log(response.routing); // { model: 'claude-sonnet-4-5', provider: 'anthropic', ... }
```

## API

### `new ModelRouter(config)`

```typescript
const ai = new ModelRouter({
  apiKey: 'mr_sk_...',           // required
  baseURL: 'https://...',        // default: https://api.lxg2it.com/v1
  defaultTier: 'standard',       // default tier for all requests
  defaultPrefer: 'balanced',     // default preference
  timeout: 60_000,               // request timeout in ms
});
```

### `ai.chat(options)`

Non-streaming chat completion.

```typescript
const response = await ai.chat({
  tier: 'standard',              // 'economy' | 'standard' | 'premium'
  prefer: 'coding',              // 'balanced' | 'cheap' | 'fast' | 'quality' | 'coding'
  include_reasoning: true,       // request chain-of-thought (where supported)
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Explain closures in JavaScript.' },
  ],
});

console.log(response.content);   // string
console.log(response.reasoning); // string | null
console.log(response.routing);   // { model, provider, tier, latency_ms }
console.log(response.usage);     // { prompt_tokens, completion_tokens, total_tokens }
```

### `ai.stream(options)`

Streaming with typed events.

```typescript
for await (const event of ai.stream({ tier: 'standard', messages: [...] })) {
  if (event.type === 'reasoning') process.stdout.write(`[thinking] ${event.delta}`);
  if (event.type === 'content')   process.stdout.write(event.delta);
  if (event.type === 'done')      console.log('\nUsed:', event.routing.model);
  if (event.type === 'error')     console.error(event.error);
}
```

### `ai.run(options)` — Agentic tool loops

```typescript
import { z } from 'zod';
import { ModelRouter, tool } from '@lxg2it/ai';

const ai = new ModelRouter({ apiKey: 'mr_sk_...' });

const getWeather = tool({
  name: 'get_weather',
  description: 'Get current weather for a city',
  parameters: z.object({ city: z.string() }),
  execute: async ({ city }) => ({ temperature: 22, condition: 'sunny', city }),
});

const result = await ai.run({
  tier: 'standard',
  messages: [{ role: 'user', content: 'What is the weather in Melbourne?' }],
  tools: [getWeather],
  maxIterations: 5,
});

console.log(result.content);        // Final response
console.log(result.toolCalls);      // All tool calls that were made
console.log(result.iterations);     // How many LLM calls were needed
console.log(result.routing.model);  // Model used for the final call
```

### `ai.embed(options)`

```typescript
const { vectors, model, usage } = await ai.embed({
  model: 'embed-small',  // 'embed-small' | 'embed-large'
  input: ['Hello world', 'How are you?'],
});
```

## Vercel AI SDK integration

```bash
npm install @lxg2it/ai ai openai
```

```typescript
import { createModelRouter } from '@lxg2it/ai/vercel';
import { streamText } from 'ai';

const mr = createModelRouter({ apiKey: 'mr_sk_...' });

// Works anywhere Vercel AI SDK accepts a model
const result = await streamText({
  model: mr('standard', { prefer: 'fast' }),
  messages: [{ role: 'user', content: 'Hello!' }],
});

// Access routing metadata
for await (const chunk of result.fullStream) {
  if (chunk.type === 'finish') {
    const meta = chunk.providerMetadata?.modelRouter;
    console.log(`Used: ${meta?.model} via ${meta?.provider}`);
  }
}
```

Works with `generateText`, `streamText`, `generateObject`, `streamObject`, `useChat`, and all other Vercel AI SDK functions.

## Routing tiers

| Tier | Use case |
|------|----------|
| `economy` | High-volume, cost-sensitive tasks (classification, extraction, simple Q&A) |
| `standard` | General purpose — the right default for most tasks |
| `premium` | Complex reasoning, long context, highest quality |

## Preferences

Use `prefer` to influence routing within a tier:

| Preference | Description |
|-----------|-------------|
| `balanced` | Default — best overall value |
| `fast` | Minimize latency |
| `cheap` | Minimize cost |
| `quality` | Maximize quality |
| `coding` | Prefer models strong on code |

## Get an API key

Sign up at [api.lxg2it.com](https://api.lxg2it.com) — free tier available.

## License

MIT
