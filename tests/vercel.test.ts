/**
 * vercel.test.ts — createModelRouter() Vercel AI SDK adapter tests.
 *
 * We test doGenerate() and doStream() directly to avoid a hard dependency on
 * the Vercel AI SDK's streamText/generateText wrappers in tests.
 */

import { describe, it, expect, afterEach } from 'vitest';
import { createModelRouter } from '../src/vercel.js';
import { makeChatResponse, makeStreamResponse, mockFetch, restoreFetch } from './helpers.js';

afterEach(() => restoreFetch());

// ─── Helper to get the model instance ────────────────────────────────────────

function getModel(tier: 'economy' | 'standard' | 'premium' = 'standard', prefer?: 'fast' | 'quality' | 'coding' | 'cheap' | 'balanced') {
  const mr = createModelRouter({ apiKey: 'test-key' });
  return mr(tier, prefer ? { prefer } : undefined);
}

const baseCallOptions = {
  mode: { type: 'regular' as const },
  prompt: [
    {
      role: 'user' as const,
      content: [{ type: 'text' as const, text: 'Hello' }],
    },
  ],
};

// ─── Provider identity ────────────────────────────────────────────────────────

describe('createModelRouter() provider identity', () => {
  it('has specificationVersion v1', () => {
    const model = getModel();
    expect(model.specificationVersion).toBe('v1');
  });

  it('has provider name "model-router"', () => {
    const model = getModel();
    expect(model.provider).toBe('model-router');
  });

  it('modelId is the tier when no prefer', () => {
    expect(getModel('economy').modelId).toBe('economy');
    expect(getModel('standard').modelId).toBe('standard');
    expect(getModel('premium').modelId).toBe('premium');
  });

  it('modelId includes prefer when specified', () => {
    expect(getModel('standard', 'coding').modelId).toBe('standard:coding');
    expect(getModel('premium', 'quality').modelId).toBe('premium:quality');
  });
});

// ─── doGenerate() ─────────────────────────────────────────────────────────────

describe('doGenerate()', () => {
  it('returns text content', async () => {
    mockFetch(makeChatResponse('Hello from the model'));
    const model = getModel();

    const result = await model.doGenerate(baseCallOptions);

    expect(result.text).toBe('Hello from the model');
  });

  it('returns usage stats', async () => {
    mockFetch(makeChatResponse('hi'));
    const model = getModel();

    const result = await model.doGenerate(baseCallOptions);

    expect(result.usage.promptTokens).toBe(10);
    expect(result.usage.completionTokens).toBe(20);
  });

  it('maps finish reason correctly', async () => {
    mockFetch(makeChatResponse('done'));
    const model = getModel();

    const result = await model.doGenerate(baseCallOptions);

    expect(result.finishReason).toBe('stop');
  });

  it('populates providerMetadata with routing info', async () => {
    mockFetch(
      makeChatResponse('hi', {
        model: 'claude-sonnet-4-5',
        provider: 'anthropic',
        tier: 'premium',
        latency_ms: '200',
      }),
    );
    const model = getModel('premium');

    const result = await model.doGenerate(baseCallOptions);

    expect(result.providerMetadata?.modelRouter).toMatchObject({
      model: 'claude-sonnet-4-5',
      provider: 'anthropic',
      tier: 'premium',
      latency_ms: 200,
    });
  });

  it('includes routing headers in rawResponse', async () => {
    mockFetch(
      makeChatResponse('hi', {
        model: 'gpt-4o',
        provider: 'openai',
        tier: 'standard',
        latency_ms: '75',
      }),
    );
    const model = getModel();

    const result = await model.doGenerate(baseCallOptions);

    expect(result.rawResponse?.headers?.['x-model-router-model']).toBe('gpt-4o');
  });

  it('sends the tier as the model param', async () => {
    const fetch = mockFetch(makeChatResponse('hi'));
    const model = getModel('economy');

    await model.doGenerate(baseCallOptions);

    const body = JSON.parse((fetch.mock.calls[0] as [string, RequestInit])[1].body as string);
    expect(body.model).toBe('economy');
  });

  it('sends prefer as extra body param', async () => {
    const fetch = mockFetch(makeChatResponse('hi'));
    const model = getModel('standard', 'fast');

    await model.doGenerate(baseCallOptions);

    const body = JSON.parse((fetch.mock.calls[0] as [string, RequestInit])[1].body as string);
    expect(body.prefer).toBe('fast');
  });
});

// ─── doStream() ────────────────────────────────────────────────────────────

describe('doStream()', () => {
  it('returns a ReadableStream', async () => {
    mockFetch(makeStreamResponse(['hi']));
    const model = getModel();

    const { stream } = await model.doStream(baseCallOptions);

    expect(stream).toBeInstanceOf(ReadableStream);
  });

  it('stream yields text-delta events', async () => {
    mockFetch(makeStreamResponse(['Hello', ' world']));
    const model = getModel();

    const { stream } = await model.doStream(baseCallOptions);
    const reader = stream.getReader();
    const events = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      events.push(value);
    }

    const textDeltas = events.filter((e) => e.type === 'text-delta');
    expect(textDeltas).toHaveLength(2);
    expect(textDeltas[0]).toMatchObject({ type: 'text-delta', textDelta: 'Hello' });
  });

  it('stream includes a finish event with routing metadata', async () => {
    mockFetch(
      makeStreamResponse(['hi'], {
        model: 'gemini-2.0-flash',
        provider: 'google',
        tier: 'standard',
        latency_ms: '90',
      }),
    );
    const model = getModel();

    const { stream } = await model.doStream(baseCallOptions);
    const reader = stream.getReader();
    const events = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      events.push(value);
    }

    const finishEvent = events.find((e) => e.type === 'finish');
    expect(finishEvent).toBeDefined();
    expect(finishEvent?.type === 'finish' && finishEvent.providerMetadata?.modelRouter).toMatchObject({
      model: 'gemini-2.0-flash',
      provider: 'google',
    });
  });

  it('returns rawCall with prompt and settings', async () => {
    mockFetch(makeStreamResponse(['hi']));
    const model = getModel();

    const { rawCall } = await model.doStream(baseCallOptions);

    expect(rawCall.rawPrompt).toBeDefined();
    expect(rawCall.rawSettings).toBeDefined();
  });
});

// ─── Prompt conversion ────────────────────────────────────────────────────────

describe('prompt conversion', () => {
  it('handles system messages', async () => {
    const fetch = mockFetch(makeChatResponse('hi'));
    const model = getModel();

    await model.doGenerate({
      ...baseCallOptions,
      prompt: [
        { role: 'system', content: 'You are helpful.' },
        { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
      ],
    });

    const body = JSON.parse((fetch.mock.calls[0] as [string, RequestInit])[1].body as string);
    expect(body.messages[0]).toMatchObject({ role: 'system', content: 'You are helpful.' });
    expect(body.messages[1]).toMatchObject({ role: 'user', content: 'Hello' });
  });

  it('handles multi-turn conversation', async () => {
    const fetch = mockFetch(makeChatResponse('hi'));
    const model = getModel();

    await model.doGenerate({
      ...baseCallOptions,
      prompt: [
        { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
        { role: 'assistant', content: [{ type: 'text', text: 'Hi there!' }] },
        { role: 'user', content: [{ type: 'text', text: 'How are you?' }] },
      ],
    });

    const body = JSON.parse((fetch.mock.calls[0] as [string, RequestInit])[1].body as string);
    expect(body.messages).toHaveLength(3);
    expect(body.messages[1]).toMatchObject({ role: 'assistant', content: 'Hi there!' });
  });

  it('converts tool result messages correctly', async () => {
    const fetch = mockFetch(makeChatResponse('hi'));
    const model = getModel();

    await model.doGenerate({
      ...baseCallOptions,
      prompt: [
        { role: 'user', content: [{ type: 'text', text: 'Weather?' }] },
        {
          role: 'assistant',
          content: [
            {
              type: 'tool-call',
              toolCallId: 'tc-1',
              toolName: 'get_weather',
              args: { city: 'Melbourne' },
            },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              type: 'tool-result',
              toolCallId: 'tc-1',
              toolName: 'get_weather',
              result: { temp: 18 },
            },
          ],
        },
      ],
    });

    const body = JSON.parse((fetch.mock.calls[0] as [string, RequestInit])[1].body as string);
    const toolMsg = body.messages.find((m: { role: string }) => m.role === 'tool');
    expect(toolMsg).toBeDefined();
    expect(toolMsg.tool_call_id).toBe('tc-1');
    expect(JSON.parse(toolMsg.content)).toEqual({ temp: 18 });
  });
});
