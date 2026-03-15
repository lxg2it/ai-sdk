/**
 * errors.test.ts — Error class and error wrapping behaviour.
 */

import { describe, it, expect } from 'vitest';
import { ModelRouterError, MaxIterationsError } from '../src/types.js';

describe('ModelRouterError', () => {
  it('has the correct name', () => {
    const err = new ModelRouterError('something went wrong');
    expect(err.name).toBe('ModelRouterError');
  });

  it('is an instance of Error', () => {
    expect(new ModelRouterError('msg')).toBeInstanceOf(Error);
  });

  it('stores status and code', () => {
    const err = new ModelRouterError('rate limited', 429, 'rate_limit_exceeded');
    expect(err.status).toBe(429);
    expect(err.code).toBe('rate_limit_exceeded');
  });

  it('status and code are optional', () => {
    const err = new ModelRouterError('generic error');
    expect(err.status).toBeUndefined();
    expect(err.code).toBeUndefined();
  });

  it('message is accessible', () => {
    const err = new ModelRouterError('bad api key');
    expect(err.message).toBe('bad api key');
  });
});

describe('MaxIterationsError', () => {
  it('has the correct name', () => {
    const err = new MaxIterationsError(5, []);
    expect(err.name).toBe('MaxIterationsError');
  });

  it('is an instance of ModelRouterError', () => {
    expect(new MaxIterationsError(3, [])).toBeInstanceOf(ModelRouterError);
  });

  it('is an instance of Error', () => {
    expect(new MaxIterationsError(3, [])).toBeInstanceOf(Error);
  });

  it('stores iterations count', () => {
    const err = new MaxIterationsError(7, []);
    expect(err.iterations).toBe(7);
  });

  it('stores partial messages', () => {
    const messages = [
      { role: 'user' as const, content: 'hello' },
      { role: 'assistant' as const, content: 'thinking...' },
    ];
    const err = new MaxIterationsError(2, messages);
    expect(err.partialMessages).toHaveLength(2);
    expect(err.partialMessages[0]).toMatchObject({ role: 'user' });
  });

  it('message includes the iteration count', () => {
    const err = new MaxIterationsError(10, []);
    expect(err.message).toContain('10');
  });
});
