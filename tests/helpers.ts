/**
 * Test helpers — mock fetch factory and response builders.
 *
 * All tests mock `globalThis.fetch` rather than mocking the OpenAI SDK,
 * so we exercise the real request-construction and header-capture paths.
 */

import { vi } from 'vitest';

// ─── Response builders ───────────────────────────────────────────────────────

export interface MockRouterHeaders {
  model?: string;
  provider?: string;
  tier?: string;
  latency_ms?: string;
}

function makeHeaders(routing: MockRouterHeaders = {}): Headers {
  const h = new Headers();
  if (routing.model) h.set('X-Model-Router-Model', routing.model);
  if (routing.provider) h.set('X-Model-Router-Provider', routing.provider);
  if (routing.tier) h.set('X-Model-Router-Tier', routing.tier);
  if (routing.latency_ms) h.set('X-Model-Router-Latency-Ms', routing.latency_ms);
  return h;
}

/** Build a fake non-streaming completion response */
export function makeChatResponse(
  content: string,
  routing: MockRouterHeaders = {},
  extra: Record<string, unknown> = {},
): Response {
  const body = JSON.stringify({
    id: 'chatcmpl-test',
    object: 'chat.completion',
    created: 1_700_000_000,
    model: routing.model ?? 'test-model',
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content, ...extra },
        finish_reason: 'stop',
      },
    ],
    usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
  });

  const headers = makeHeaders(routing);
  headers.set('Content-Type', 'application/json');
  return new Response(body, { status: 200, headers });
}

/** Build a fake completion response with tool_calls */
export function makeToolCallResponse(
  toolCalls: Array<{ id: string; name: string; arguments: string }>,
  routing: MockRouterHeaders = {},
): Response {
  const body = JSON.stringify({
    id: 'chatcmpl-tool',
    object: 'chat.completion',
    created: 1_700_000_000,
    model: routing.model ?? 'test-model',
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: null,
          tool_calls: toolCalls.map((tc) => ({
            id: tc.id,
            type: 'function',
            function: { name: tc.name, arguments: tc.arguments },
          })),
        },
        finish_reason: 'tool_calls',
      },
    ],
    usage: { prompt_tokens: 15, completion_tokens: 25, total_tokens: 40 },
  });

  const headers = makeHeaders(routing);
  headers.set('Content-Type', 'application/json');
  return new Response(body, { status: 200, headers });
}

/** Build a fake embeddings response */
export function makeEmbedResponse(
  vectors: number[][],
  routing: MockRouterHeaders = {},
): Response {
  const body = JSON.stringify({
    object: 'list',
    data: vectors.map((embedding, index) => ({ object: 'embedding', index, embedding })),
    model: 'embed-small',
    usage: { prompt_tokens: 5, total_tokens: 5 },
  });

  const headers = makeHeaders(routing);
  headers.set('Content-Type', 'application/json');
  return new Response(body, { status: 200, headers });
}

/** Build an error response (e.g. 401, 429, 500) */
export function makeErrorResponse(status: number, message: string): Response {
  const body = JSON.stringify({
    error: { message, type: 'error', code: String(status) },
  });
  return new Response(body, {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

// ─── SSE stream helpers ──────────────────────────────────────────────────────

/** Encode a single SSE data line */
function sseChunk(data: unknown): Uint8Array {
  return new TextEncoder().encode(`data: ${JSON.stringify(data)}\n\n`);
}

function sseDone(): Uint8Array {
  return new TextEncoder().encode('data: [DONE]\n\n');
}

/**
 * Build a fake streaming response from an array of content deltas.
 * Mirrors the OpenAI SSE format closely enough for the SDK to parse.
 */
export function makeStreamResponse(
  deltas: string[],
  routing: MockRouterHeaders = {},
): Response {
  const chunks: Uint8Array[] = [];

  // Content delta chunks
  for (const delta of deltas) {
    chunks.push(
      sseChunk({
        id: 'chatcmpl-stream',
        object: 'chat.completion.chunk',
        created: 1_700_000_000,
        model: routing.model ?? 'test-model',
        choices: [{ index: 0, delta: { content: delta }, finish_reason: null }],
      }),
    );
  }

  // Final chunk with finish_reason and usage
  chunks.push(
    sseChunk({
      id: 'chatcmpl-stream',
      object: 'chat.completion.chunk',
      created: 1_700_000_000,
      model: routing.model ?? 'test-model',
      choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
      usage: { prompt_tokens: 10, completion_tokens: deltas.length * 2, total_tokens: 10 + deltas.length * 2 },
    }),
  );

  chunks.push(sseDone());

  const stream = new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(chunk);
      }
      controller.close();
    },
  });

  return new Response(stream, {
    status: 200,
    headers: {
      'Content-Type': 'text/event-stream',
      ...Object.fromEntries(makeHeaders(routing).entries()),
    },
  });
}

// ─── Mock fetch setup ────────────────────────────────────────────────────────

/** Replace globalThis.fetch with a mock that returns responses from a queue */
export function mockFetch(...responses: Response[]): ReturnType<typeof vi.fn> {
  const queue = [...responses];
  const mock = vi.fn().mockImplementation(() => {
    const next = queue.shift();
    if (!next) throw new Error('mockFetch: no more responses queued');
    return Promise.resolve(next);
  });
  vi.stubGlobal('fetch', mock);
  return mock;
}

/** Restore globalThis.fetch after mocking */
export function restoreFetch(): void {
  vi.unstubAllGlobals();
}
