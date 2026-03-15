/**
 * @lxg2it/ai — Shared utilities
 */

// ─── Header capture ──────────────────────────────────────────────────────────

export type CapturedHeaders = {
  model: string | null;
  provider: string | null;
  tier: string | null;
  latency_ms: string | null;
};

/**
 * Creates a fetch wrapper that captures X-Model-Router-* response headers into
 * a closure-scoped store. The OpenAI SDK doesn't expose HTTP response headers in
 * its typed return values, so we intercept them here for routing metadata.
 */
export function createHeaderCapturingFetch(): {
  fetch: typeof globalThis.fetch;
  getLastHeaders: () => CapturedHeaders;
} {
  let lastHeaders: CapturedHeaders = {
    model: null,
    provider: null,
    tier: null,
    latency_ms: null,
  };

  const wrappedFetch = async (
    input: Parameters<typeof globalThis.fetch>[0],
    init?: RequestInit,
  ): Promise<Response> => {
    const response = await globalThis.fetch(input, init);

    lastHeaders = {
      model: response.headers.get('X-Model-Router-Model'),
      provider: response.headers.get('X-Model-Router-Provider'),
      tier: response.headers.get('X-Model-Router-Tier'),
      latency_ms: response.headers.get('X-Model-Router-Latency-Ms'),
    };

    return response;
  };

  return {
    fetch: wrappedFetch,
    getLastHeaders: () => ({ ...lastHeaders }),
  };
}

/**
 * Converts a raw header value to the Vercel AI SDK's expected flat header
 * record (string values only, nulls omitted).
 */
export function headersToRecord(
  headers: CapturedHeaders,
): Record<string, string> {
  return Object.fromEntries(
    [
      ['x-model-router-model', headers.model],
      ['x-model-router-provider', headers.provider],
      ['x-model-router-tier', headers.tier],
      ['x-model-router-latency-ms', headers.latency_ms],
    ].filter((entry): entry is [string, string] => entry[1] !== null),
  );
}
