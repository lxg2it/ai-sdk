/**
 * tool.test.ts — tool() factory function tests.
 */

import { describe, it, expect } from 'vitest';
import { z } from 'zod';
import { tool } from '../src/client.js';

// ─── tool() factory ──────────────────────────────────────────────────────────

describe('tool()', () => {
  it('preserves name and description', () => {
    const t = tool({
      name: 'get_weather',
      description: 'Get the weather for a city',
      parameters: z.object({ city: z.string() }),
      execute: async () => ({}),
    });

    expect(t.name).toBe('get_weather');
    expect(t.description).toBe('Get the weather for a city');
  });

  it('generates a JSON schema from a Zod schema', () => {
    const t = tool({
      name: 'search',
      parameters: z.object({
        query: z.string().describe('The search query'),
        limit: z.number().int().min(1).max(100).optional(),
      }),
      execute: async () => [],
    });

    const schema = t.jsonSchema as {
      type: string;
      properties: Record<string, unknown>;
      required?: string[];
    };

    expect(schema.type).toBe('object');
    // query should be present and typed as string
    expect(schema.properties['query']).toMatchObject({ type: 'string' });
    // limit should be present (optional() still generates a property, just with nullable union)
    expect(schema.properties).toHaveProperty('limit');
    // query is required
    expect(schema.required).toContain('query');
  });

  it('falls back to empty schema when not a Zod schema', () => {
    const plainParser = {
      parse: (input: unknown) => input as { city: string },
      // No _def property — not a Zod schema
    };

    const t = tool({
      name: 'get_weather',
      parameters: plainParser,
      execute: async () => ({}),
    });

    expect(t.jsonSchema).toEqual({ type: 'object', properties: {} });
  });

  it('execute function is preserved and callable', async () => {
    const t = tool({
      name: 'add',
      parameters: z.object({ a: z.number(), b: z.number() }),
      execute: async ({ a, b }) => a + b,
    });

    const result = await t.execute({ a: 3, b: 4 });
    expect(result).toBe(7);
  });

  it('parameters.parse validates input', () => {
    const t = tool({
      name: 'test',
      parameters: z.object({ value: z.number() }),
      execute: async () => null,
    });

    expect(() => t.parameters.parse({ value: 'not-a-number' })).toThrow();
    expect(() => t.parameters.parse({ value: 42 })).not.toThrow();
  });

  it('handles undefined description gracefully', () => {
    const t = tool({
      name: 'nodesc',
      parameters: z.object({}),
      execute: async () => null,
    });

    expect(t.description).toBeUndefined();
  });

  it('works with nested Zod schemas', () => {
    const t = tool({
      name: 'create_order',
      parameters: z.object({
        customer: z.object({
          name: z.string(),
          email: z.string().email(),
        }),
        items: z.array(
          z.object({ sku: z.string(), qty: z.number().int().positive() }),
        ),
      }),
      execute: async () => ({ orderId: '123' }),
    });

    expect(t.jsonSchema).toMatchObject({
      type: 'object',
      properties: {
        customer: expect.objectContaining({ type: 'object' }),
        items: expect.objectContaining({ type: 'array' }),
      },
    });
  });
});
