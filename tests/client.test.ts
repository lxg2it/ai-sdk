/**
 * client.test.ts — ModelRouter chat(), embed(), and config tests.
 *
 * Streaming and run() have their own test files to keep things focused.
 */

import { describe, it, expect, afterEach } from 'vitest';
import { ModelRouter } from '../src/client.js';
import { ModelRouterError } from '../src/types.js';
import {
  makeChatResponse,
  makeEmbedResponse,
  makeErrorResponse,
  mockFetch,
  restoreFetch,
} from './helpers.js';

afterEach(() => restoreFetch());

// ─── Constructor / config ────────────────────────────────────────────────────

describe('ModelRouter config', () => {
  it('applies default tier and prefer when not specified', async () => {
    const fetch = mockFetch(makeChatResponse('hello'));
    const ai = new ModelRouter({ apiKey: 'test-key' });

    await ai.chat({ messages: [{ role: 'user', content: 'hi' }] });

    const body = JSON.parse((fetch.mock.calls[0] as [string, RequestInit])[1].body as string);
    expect(body.model).toBe('standard');
    expect(body.prefer).toBe('balanced');
  });

  it('respects custom defaultTier and defaultPrefer', async () => {
    const fetch = mockFetch(makeChatResponse('hello'));
    const ai = new ModelRouter({ apiKey: 'key', defaultTier: 'premium', defaultPrefer: 'coding' });

    await ai.chat({ messages: [{ role: 'user', content: 'hi' }] });

    const body = JSON.parse((fetch.mock.calls[0] as [string, RequestInit])[1].body as string);
    expect(body.model).toBe('premium');
    expect(body.prefer).toBe('coding');
  });

  it('uses the default base URL', async () => {
    const fetch = mockFetch(makeChatResponse('hello'));
    const ai = new ModelRouter({ apiKey: 'key' });

    await ai.chat({ messages: [{ role: 'user', content: 'hi' }] });

    const url = (fetch.mock.calls[0] as [string])[0];
    expect(url).toContain('api.lxg2it.com');
  });

  it('accepts a custom baseURL', async () => {
    const fetch = mockFetch(makeChatResponse('hello'));
    const ai = new ModelRouter({ apiKey: 'key', baseURL: 'https://my-router.example.com/v1' });

    await ai.chat({ messages: [{ role: 'user', content: 'hi' }] });

    const url = (fetch.mock.calls[0] as [string])[0];
    expect(url).toContain('my-router.example.com');
  });
});

// ─── chat() ─────────────────────────────────────────────────────────────────

describe('ModelRouter.chat()', () => {
  it('returns content from the response', async () => {
    mockFetch(makeChatResponse('Hello world'));
    const ai = new ModelRouter({ apiKey: 'key' });

    const result = await ai.chat({ messages: [{ role: 'user', content: 'hi' }] });

    expect(result.content).toBe('Hello world');
  });

  it('populates routing metadata from response headers', async () => {
    mockFetch(
      makeChatResponse('hi', {
        model: 'claude-sonnet-4-5',
        provider: 'anthropic',
        tier: 'premium',
        latency_ms: '342',
      }),
    );
    const ai = new ModelRouter({ apiKey: 'key' });

    const result = await ai.chat({ messages: [{ role: 'user', content: 'hi' }] });

    expect(result.routing.model).toBe('claude-sonnet-4-5');
    expect(result.routing.provider).toBe('anthropic');
    expect(result.routing.tier).toBe('premium');
    expect(result.routing.latency_ms).toBe(342);
  });

  it('returns null routing fields as defaults when headers are absent', async () => {
    mockFetch(makeChatResponse('hi'));
    const ai = new ModelRouter({ apiKey: 'key' });

    const result = await ai.chat({ messages: [{ role: 'user', content: 'hi' }] });

    expect(result.routing.model).toBe('unknown');
    expect(result.routing.provider).toBe('unknown');
  });

  it('includes usage info', async () => {
    mockFetch(makeChatResponse('hi'));
    const ai = new ModelRouter({ apiKey: 'key' });

    const result = await ai.chat({ messages: [{ role: 'user', content: 'hi' }] });

    expect(result.usage).toEqual({ prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 });
  });

  it('per-request tier and prefer override defaults', async () => {
    const fetch = mockFetch(makeChatResponse('hi'));
    const ai = new ModelRouter({ apiKey: 'key', defaultTier: 'standard', defaultPrefer: 'fast' });

    await ai.chat({
      messages: [{ role: 'user', content: 'hi' }],
      tier: 'economy',
      prefer: 'cheap',
    });

    const body = JSON.parse((fetch.mock.calls[0] as [string, RequestInit])[1].body as string);
    expect(body.model).toBe('economy');
    expect(body.prefer).toBe('cheap');
  });

  it('sends per-request apiKey override in Authorization header', async () => {
    const fetch = mockFetch(makeChatResponse('hi'));
    const ai = new ModelRouter({ apiKey: 'default-key' });

    await ai.chat({
      messages: [{ role: 'user', content: 'hi' }],
      apiKey: 'override-key',
    });

    const headers = (fetch.mock.calls[0] as [string, RequestInit])[1].headers as Record<string, string>;
    expect(headers['authorization']).toBe('Bearer override-key');
  });

  it('uses the default apiKey when no override is supplied', async () => {
    const fetch = mockFetch(makeChatResponse('hi'));
    const ai = new ModelRouter({ apiKey: 'default-key' });

    await ai.chat({ messages: [{ role: 'user', content: 'hi' }] });

    const headers = (fetch.mock.calls[0] as [string, RequestInit])[1].headers as Record<string, string>;
    expect(headers['authorization']).toBe('Bearer default-key');
  });

  it('returns reasoning_content when present', async () => {
    mockFetch(makeChatResponse('answer', {}, { reasoning_content: 'I thought about it...' }));
    const ai = new ModelRouter({ apiKey: 'key' });

    const result = await ai.chat({
      messages: [{ role: 'user', content: 'think' }],
      include_reasoning: true,
    });

    expect(result.reasoning).toBe('I thought about it...');
  });

  it('returns null reasoning when not present', async () => {
    mockFetch(makeChatResponse('answer'));
    const ai = new ModelRouter({ apiKey: 'key' });

    const result = await ai.chat({ messages: [{ role: 'user', content: 'hi' }] });

    expect(result.reasoning).toBeNull();
  });

  it('throws ModelRouterError on 401', async () => {
    mockFetch(makeErrorResponse(401, 'Unauthorized'));
    const ai = new ModelRouter({ apiKey: 'bad-key', maxRetries: 0 });

    await expect(
      ai.chat({ messages: [{ role: 'user', content: 'hi' }] }),
    ).rejects.toBeInstanceOf(ModelRouterError);
  });

  it('includes HTTP status in ModelRouterError', async () => {
    mockFetch(makeErrorResponse(429, 'Rate limited'));
    const ai = new ModelRouter({ apiKey: 'key', maxRetries: 0 });

    const err = await ai
      .chat({ messages: [{ role: 'user', content: 'hi' }] })
      .catch((e: unknown) => e);

    expect(err).toBeInstanceOf(ModelRouterError);
    expect((err as ModelRouterError).status).toBe(429);
  });
});

// ─── embed() ────────────────────────────────────────────────────────────────

describe('ModelRouter.embed()', () => {
  it('returns embedding vectors', async () => {
    const vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
    mockFetch(makeEmbedResponse(vectors));
    const ai = new ModelRouter({ apiKey: 'key' });

    const result = await ai.embed({ input: ['hello', 'world'] });

    expect(result.vectors).toEqual(vectors);
    expect(result.vectors).toHaveLength(2);
  });

  it('returns model and usage info', async () => {
    mockFetch(makeEmbedResponse([[0.1, 0.2]]));
    const ai = new ModelRouter({ apiKey: 'key' });

    const result = await ai.embed({ input: 'test' });

    expect(result.model).toBe('embed-small');
    expect(result.usage.prompt_tokens).toBe(5);
    expect(result.usage.total_tokens).toBe(5);
  });

  it('uses embed-small model by default', async () => {
    const fetch = mockFetch(makeEmbedResponse([[0.1]]));
    const ai = new ModelRouter({ apiKey: 'key' });

    await ai.embed({ input: 'test' });

    const body = JSON.parse((fetch.mock.calls[0] as [string, RequestInit])[1].body as string);
    expect(body.model).toBe('embed-small');
  });

  it('accepts model override', async () => {
    const fetch = mockFetch(makeEmbedResponse([[0.1]]));
    const ai = new ModelRouter({ apiKey: 'key' });

    await ai.embed({ input: 'test', model: 'embed-large' });

    const body = JSON.parse((fetch.mock.calls[0] as [string, RequestInit])[1].body as string);
    expect(body.model).toBe('embed-large');
  });

  it('throws ModelRouterError on API error', async () => {
    mockFetch(makeErrorResponse(500, 'Internal server error'));
    const ai = new ModelRouter({ apiKey: 'key', maxRetries: 0 });

    await expect(ai.embed({ input: 'test' })).rejects.toBeInstanceOf(ModelRouterError);
  });
});
