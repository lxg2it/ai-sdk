/**
 * stream.test.ts — ModelRouter.stream() tests.
 *
 * Uses SSE mock responses to exercise the full streaming path including
 * content deltas, tool calls, routing metadata, and error events.
 */

import { describe, it, expect, afterEach } from 'vitest';
import { ModelRouter } from '../src/client.js';
import { ModelRouterError } from '../src/types.js';
import type { StreamEvent } from '../src/types.js';
import { makeStreamResponse, makeErrorResponse, mockFetch, restoreFetch } from './helpers.js';

afterEach(() => restoreFetch());

async function collectEvents(gen: AsyncGenerator<StreamEvent>): Promise<StreamEvent[]> {
  const events: StreamEvent[] = [];
  for await (const event of gen) {
    events.push(event);
  }
  return events;
}

// ─── Basic streaming ─────────────────────────────────────────────────────────

describe('ModelRouter.stream()', () => {
  it('yields content delta events', async () => {
    mockFetch(makeStreamResponse(['Hello', ' world']));
    const ai = new ModelRouter({ apiKey: 'key' });

    const events = await collectEvents(
      ai.stream({ messages: [{ role: 'user', content: 'hi' }] }),
    );

    const contentEvents = events.filter((e) => e.type === 'content');
    expect(contentEvents).toHaveLength(2);
    expect(contentEvents[0]).toMatchObject({ type: 'content', delta: 'Hello' });
    expect(contentEvents[1]).toMatchObject({ type: 'content', delta: ' world' });
  });

  it('yields a done event at the end', async () => {
    mockFetch(makeStreamResponse(['hi']));
    const ai = new ModelRouter({ apiKey: 'key' });

    const events = await collectEvents(
      ai.stream({ messages: [{ role: 'user', content: 'hello' }] }),
    );

    const doneEvents = events.filter((e) => e.type === 'done');
    expect(doneEvents).toHaveLength(1);
  });

  it('done event contains routing metadata', async () => {
    mockFetch(
      makeStreamResponse(['hi'], {
        model: 'gpt-4o',
        provider: 'openai',
        tier: 'standard',
        latency_ms: '150',
      }),
    );
    const ai = new ModelRouter({ apiKey: 'key' });

    const events = await collectEvents(
      ai.stream({ messages: [{ role: 'user', content: 'hello' }] }),
    );

    const done = events.find((e) => e.type === 'done');
    expect(done).toBeDefined();
    if (done?.type === 'done') {
      expect(done.routing.model).toBe('gpt-4o');
      expect(done.routing.provider).toBe('openai');
      expect(done.routing.latency_ms).toBe(150);
    }
  });

  it('done event contains usage info', async () => {
    mockFetch(makeStreamResponse(['a', 'b', 'c']));
    const ai = new ModelRouter({ apiKey: 'key' });

    const events = await collectEvents(
      ai.stream({ messages: [{ role: 'user', content: 'hi' }] }),
    );

    const done = events.find((e) => e.type === 'done');
    expect(done).toBeDefined();
    expect(done?.type).toBe('done');
    if (done?.type === 'done') {
      expect(done.usage).not.toBeNull();
      expect(done.usage?.prompt_tokens).toBe(10);
    }
  });

  it('uses per-request tier and prefer', async () => {
    const fetch = mockFetch(makeStreamResponse(['hi']));
    const ai = new ModelRouter({ apiKey: 'key' });

    await collectEvents(
      ai.stream({
        messages: [{ role: 'user', content: 'hi' }],
        tier: 'premium',
        prefer: 'quality',
      }),
    );

    const body = JSON.parse((fetch.mock.calls[0] as [string, RequestInit])[1].body as string);
    expect(body.model).toBe('premium');
    expect(body.prefer).toBe('quality');
  });

  it('yields an error event on API failure', async () => {
    mockFetch(makeErrorResponse(500, 'Internal server error'));
    const ai = new ModelRouter({ apiKey: 'key', maxRetries: 0 });

    const events = await collectEvents(
      ai.stream({ messages: [{ role: 'user', content: 'hi' }] }),
    );

    const errorEvents = events.filter((e) => e.type === 'error');
    expect(errorEvents).toHaveLength(1);
    expect(errorEvents[0]).toMatchObject({ type: 'error' });
    if (errorEvents[0].type === 'error') {
      expect(errorEvents[0].error).toBeInstanceOf(ModelRouterError);
    }
  });

  it('error event wraps HTTP status', async () => {
    mockFetch(makeErrorResponse(401, 'Unauthorized'));
    const ai = new ModelRouter({ apiKey: 'bad-key', maxRetries: 0 });

    const events = await collectEvents(
      ai.stream({ messages: [{ role: 'user', content: 'hi' }] }),
    );

    const errorEvent = events.find((e) => e.type === 'error');
    expect(errorEvent?.type).toBe('error');
    if (errorEvent?.type === 'error') {
      expect((errorEvent.error as ModelRouterError).status).toBe(401);
    }
  });

  it('content events can be reassembled into the full response', async () => {
    mockFetch(makeStreamResponse(['The ', 'answer ', 'is 42']));
    const ai = new ModelRouter({ apiKey: 'key' });

    const events = await collectEvents(
      ai.stream({ messages: [{ role: 'user', content: 'what is the answer?' }] }),
    );

    const fullText = events
      .filter((e) => e.type === 'content')
      .map((e) => (e.type === 'content' ? e.delta : ''))
      .join('');

    expect(fullText).toBe('The answer is 42');
  });
});
