/**
 * run.test.ts — ModelRouter.run() agentic tool-use loop tests.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import { z } from 'zod';
import { ModelRouter, tool } from '../src/client.js';
import { ModelRouterError, MaxIterationsError } from '../src/types.js';
import { makeChatResponse, makeToolCallResponse, makeErrorResponse, mockFetch, restoreFetch } from './helpers.js';

afterEach(() => {
  restoreFetch();
  vi.restoreAllMocks();
});

// ─── Happy path ──────────────────────────────────────────────────────────────

describe('ModelRouter.run()', () => {
  it('returns content directly when no tool calls are made', async () => {
    mockFetch(makeChatResponse('The weather is sunny', { model: 'gpt-4o', provider: 'openai', tier: 'standard', latency_ms: '100' }));
    const ai = new ModelRouter({ apiKey: 'key' });

    const getWeather = tool({
      name: 'get_weather',
      description: 'Get weather',
      parameters: z.object({ city: z.string() }),
      execute: async ({ city }) => ({ city, temp: 22 }),
    });

    const result = await ai.run({
      messages: [{ role: 'user', content: 'What is the weather?' }],
      tools: [getWeather],
    });

    expect(result.content).toBe('The weather is sunny');
    expect(result.toolCalls).toHaveLength(0);
    expect(result.iterations).toBe(1);
  });

  it('executes a tool call and sends the result back', async () => {
    const weatherExecute = vi.fn().mockResolvedValue({ city: 'Melbourne', temp: 18 });

    mockFetch(
      makeToolCallResponse([{ id: 'tc-1', name: 'get_weather', arguments: '{"city":"Melbourne"}' }], {
        model: 'gpt-4o',
        provider: 'openai',
        tier: 'standard',
        latency_ms: '80',
      }),
      makeChatResponse('It is 18°C in Melbourne.', { model: 'gpt-4o', provider: 'openai', tier: 'standard', latency_ms: '60' }),
    );

    const ai = new ModelRouter({ apiKey: 'key' });

    const getWeather = tool({
      name: 'get_weather',
      description: 'Get weather',
      parameters: z.object({ city: z.string() }),
      execute: weatherExecute,
    });

    const result = await ai.run({
      messages: [{ role: 'user', content: 'Weather in Melbourne?' }],
      tools: [getWeather],
    });

    expect(weatherExecute).toHaveBeenCalledWith({ city: 'Melbourne' });
    expect(result.content).toBe('It is 18°C in Melbourne.');
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0].name).toBe('get_weather');
    expect(result.toolCalls[0].result).toEqual({ city: 'Melbourne', temp: 18 });
    expect(result.iterations).toBe(2);
  });

  it('aggregates usage across multiple iterations', async () => {
    mockFetch(
      makeToolCallResponse([{ id: 'tc-1', name: 'get_weather', arguments: '{"city":"Sydney"}' }]),
      makeChatResponse('Done'),
    );

    const ai = new ModelRouter({ apiKey: 'key' });

    const result = await ai.run({
      messages: [{ role: 'user', content: 'hi' }],
      tools: [
        tool({
          name: 'get_weather',
          description: 'Get weather',
          parameters: z.object({ city: z.string() }),
          execute: async () => ({ temp: 25 }),
        }),
      ],
    });

    // First call: prompt=15, completion=25 → total=40
    // Second call: prompt=10, completion=20 → total=30
    expect(result.usage.total_tokens).toBe(70);
  });

  it('includes the updated messages array in the result', async () => {
    mockFetch(
      makeToolCallResponse([{ id: 'tc-1', name: 'get_weather', arguments: '{"city":"Perth"}' }]),
      makeChatResponse('Warm in Perth.'),
    );

    const ai = new ModelRouter({ apiKey: 'key' });

    const result = await ai.run({
      messages: [{ role: 'user', content: 'Weather?' }],
      tools: [
        tool({
          name: 'get_weather',
          description: '',
          parameters: z.object({ city: z.string() }),
          execute: async () => ({ temp: 30 }),
        }),
      ],
    });

    // user + assistant (tool_calls) + tool result + assistant (final)
    expect(result.messages.length).toBeGreaterThanOrEqual(4);
    expect(result.messages[0]).toMatchObject({ role: 'user' });
  });

  it('returns an error result when a tool is not found', async () => {
    const fetch = mockFetch(
      makeToolCallResponse([{ id: 'tc-1', name: 'unknown_tool', arguments: '{}' }]),
      makeChatResponse("I couldn't find that tool."),
    );

    const ai = new ModelRouter({ apiKey: 'key' });

    const result = await ai.run({
      messages: [{ role: 'user', content: 'hi' }],
      tools: [],
    });

    // The second fetch (after sending tool error back) should be called
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(result.toolCalls[0].result).toMatchObject({ error: expect.stringContaining('unknown_tool') });
  });

  it('routing metadata comes from the final LLM call', async () => {
    mockFetch(
      makeToolCallResponse([{ id: 'tc-1', name: 'get_weather', arguments: '{"city":"Hobart"}' }], {
        model: 'gpt-4o-mini',
        provider: 'openai',
        tier: 'economy',
        latency_ms: '50',
      }),
      makeChatResponse('Cold in Hobart.', {
        model: 'claude-haiku',
        provider: 'anthropic',
        tier: 'standard',
        latency_ms: '120',
      }),
    );

    const ai = new ModelRouter({ apiKey: 'key' });

    const result = await ai.run({
      messages: [{ role: 'user', content: 'hi' }],
      tools: [
        tool({
          name: 'get_weather',
          description: '',
          parameters: z.object({ city: z.string() }),
          execute: async () => ({ temp: 12 }),
        }),
      ],
    });

    // Should be from the second call (final response)
    expect(result.routing.model).toBe('claude-haiku');
    expect(result.routing.provider).toBe('anthropic');
  });
});

// ─── maxIterations ───────────────────────────────────────────────────────────

describe('ModelRouter.run() maxIterations', () => {
  it('throws MaxIterationsError by default when limit is reached', async () => {
    // Always respond with a tool call → infinite loop guard
    mockFetch(
      makeToolCallResponse([{ id: 'tc-1', name: 'get_weather', arguments: '{"city":"X"}' }]),
      makeToolCallResponse([{ id: 'tc-2', name: 'get_weather', arguments: '{"city":"X"}' }]),
      makeToolCallResponse([{ id: 'tc-3', name: 'get_weather', arguments: '{"city":"X"}' }]),
    );

    const ai = new ModelRouter({ apiKey: 'key' });

    const err = await ai
      .run({
        messages: [{ role: 'user', content: 'hi' }],
        tools: [
          tool({
            name: 'get_weather',
            description: '',
            parameters: z.object({ city: z.string() }),
            execute: async () => ({ temp: 0 }),
          }),
        ],
        maxIterations: 3,
      })
      .catch((e: unknown) => e);

    expect(err).toBeInstanceOf(MaxIterationsError);
    expect((err as MaxIterationsError).iterations).toBe(3);
  });

  it('returns partial result when onMaxIterations is "return"', async () => {
    mockFetch(
      makeToolCallResponse([{ id: 'tc-1', name: 'get_weather', arguments: '{"city":"X"}' }]),
      makeToolCallResponse([{ id: 'tc-2', name: 'get_weather', arguments: '{"city":"X"}' }]),
    );

    const ai = new ModelRouter({ apiKey: 'key' });

    const result = await ai.run({
      messages: [{ role: 'user', content: 'hi' }],
      tools: [
        tool({
          name: 'get_weather',
          description: '',
          parameters: z.object({ city: z.string() }),
          execute: async () => ({ temp: 0 }),
        }),
      ],
      maxIterations: 2,
      onMaxIterations: 'return',
    });

    expect(result.iterations).toBe(2);
    expect(result.toolCalls.length).toBeGreaterThan(0);
  });

  it('MaxIterationsError carries the partial messages', async () => {
    mockFetch(
      makeToolCallResponse([{ id: 'tc-1', name: 'get_weather', arguments: '{"city":"X"}' }]),
    );

    const ai = new ModelRouter({ apiKey: 'key' });

    const err = await ai
      .run({
        messages: [{ role: 'user', content: 'hi' }],
        tools: [
          tool({
            name: 'get_weather',
            description: '',
            parameters: z.object({ city: z.string() }),
            execute: async () => ({ temp: 0 }),
          }),
        ],
        maxIterations: 1,
      })
      .catch((e: unknown) => e);

    expect(err).toBeInstanceOf(MaxIterationsError);
    expect((err as MaxIterationsError).partialMessages.length).toBeGreaterThan(0);
  });
});

// ─── Error propagation ───────────────────────────────────────────────────────

describe('ModelRouter.run() error handling', () => {
  it('throws ModelRouterError on API error', async () => {
    mockFetch(makeErrorResponse(500, 'Server error'));
    const ai = new ModelRouter({ apiKey: 'key', maxRetries: 0 });

    await expect(
      ai.run({
        messages: [{ role: 'user', content: 'hi' }],
        tools: [],
      }),
    ).rejects.toBeInstanceOf(ModelRouterError);
  });
});
