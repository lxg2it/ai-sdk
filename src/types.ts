/**
 * @lxg2it/ai — TypeScript types
 *
 * The public API surface for the Model Router SDK.
 * All types used in the client API are defined here.
 */

// ─── Core Router Types ──────────────────────────────────────────────────────

/** Routing tiers — defines the capability pool to draw from */
export type Tier = 'economy' | 'standard' | 'premium';

/** Routing preference — selects within the tier pool */
export type Prefer = 'balanced' | 'cheap' | 'fast' | 'quality' | 'coding';

/** Embedding model identifiers */
export type EmbeddingModel = 'embed-small' | 'embed-large';

/** Provider names as reported by the router */
export type Provider = 'anthropic' | 'openai' | 'google' | 'grok' | 'bedrock';

// ─── Message Types ──────────────────────────────────────────────────────────

export interface SystemMessage {
  role: 'system';
  content: string;
}

export interface UserMessage {
  role: 'user';
  content: string | ContentPart[];
}

export interface AssistantMessage {
  role: 'assistant';
  content: string;
  reasoning?: string;
  tool_calls?: ToolCall[];
}

export interface ToolResultMessage {
  role: 'tool';
  tool_call_id: string;
  content: string;
}

export type Message = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage;

export interface ContentPart {
  type: 'text' | 'image_url';
  text?: string;
  image_url?: { url: string; detail?: 'auto' | 'low' | 'high' };
}

export interface ToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}

// ─── Routing Metadata ──────────────────────────────────────────────────────

/**
 * Routing metadata extracted from X-Model-Router-* response headers.
 * Surfaces which model and provider actually handled the request.
 */
export interface RoutingMetadata {
  /** The actual model that handled the request, e.g. "claude-sonnet-4-5" */
  model: string;
  /** The provider that handled the request */
  provider: Provider;
  /** The tier used for routing */
  tier: Tier;
  /** Time-to-first-token in milliseconds as measured by the router */
  latency_ms: number;
}

// ─── Usage ─────────────────────────────────────────────────────────────────

export interface UsageInfo {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

// ─── chat() types ──────────────────────────────────────────────────────────

export interface ChatOptions {
  messages: Message[];
  tier?: Tier;
  prefer?: Prefer;
  include_reasoning?: boolean;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop?: string | string[];
  response_format?: { type: 'text' | 'json_object' };
  /** Override the API key for this request only */
  apiKey?: string;
}

export interface ChatResponse {
  /** The assistant's reply text */
  content: string;
  /** Chain-of-thought content when include_reasoning: true and available */
  reasoning: string | null;
  /** Routing metadata — which model/provider actually handled this */
  routing: RoutingMetadata;
  /** Token usage */
  usage: UsageInfo | null;
}

// ─── stream() types ────────────────────────────────────────────────────────

export type StreamOptions = Omit<ChatOptions, 'response_format'> & {
  response_format?: { type: 'text' | 'json_object' };
};

/** Typed events yielded by stream() */
export type StreamEvent =
  /** A chunk of reasoning/thinking content */
  | { type: 'reasoning'; delta: string }
  /** A chunk of the main response content */
  | { type: 'content'; delta: string }
  /** A partial tool call argument delta (being assembled) */
  | { type: 'tool_call_delta'; id: string; name: string; argumentsDelta: string }
  /** A complete, fully-assembled tool call */
  | { type: 'tool_call'; id: string; name: string; arguments: string }
  /** Stream completed — routing metadata is available here */
  | {
      type: 'done';
      routing: RoutingMetadata;
      usage: UsageInfo | null;
    }
  /** An error occurred — stream ends after this event */
  | { type: 'error'; error: Error };

// ─── tool() types ──────────────────────────────────────────────────────────

/**
 * A tool definition with a Zod schema for type-safe argument parsing.
 *
 * Create tools using the tool() factory function:
 *
 * ```typescript
 * import { z } from 'zod';
 * import { tool } from '@lxg2it/ai';
 *
 * const getWeather = tool({
 *   name: 'get_weather',
 *   description: 'Get current weather for a city',
 *   parameters: z.object({ city: z.string() }),
 *   execute: async ({ city }) => ({ temperature: 22, condition: 'sunny' }),
 * });
 * ```
 */
export interface ToolDefinition<TParams = Record<string, unknown>, TResult = unknown> {
  name: string;
  description?: string;
  /** Zod schema for parameter validation and TypeScript inference */
  parameters: {
    parse: (input: unknown) => TParams;
    _def?: unknown; // Zod internal — used for JSON schema conversion
  };
  execute: (params: TParams) => Promise<TResult>;
  /** JSON schema representation (generated from parameters) */
  readonly jsonSchema: Record<string, unknown>;
}

// ─── run() types ───────────────────────────────────────────────────────────

export interface RunOptions extends Omit<ChatOptions, 'tools' | 'tool_choice'> {
  tools: ToolDefinition[];
  /** Stop the loop after this many LLM calls. Default: 10 */
  maxIterations?: number;
  /** What to do when maxIterations is reached: 'error' | 'return'. Default: 'error' */
  onMaxIterations?: 'error' | 'return';
}

export interface ToolCallRecord {
  name: string;
  args: Record<string, unknown>;
  result: unknown;
}

export interface RunResponse {
  /** The final assistant response after all tool calls completed */
  content: string;
  /** All tool calls that were executed, in order */
  toolCalls: ToolCallRecord[];
  /** Number of LLM calls made (including the final one) */
  iterations: number;
  /** Routing metadata from the final LLM call */
  routing: RoutingMetadata;
  /** Aggregate token usage across all iterations */
  usage: UsageInfo;
  /**
   * Full updated messages array including all tool calls and results.
   * Useful for continuing the conversation or saving state.
   */
  messages: Message[];
}

// ─── embed() types ─────────────────────────────────────────────────────────

export interface EmbedOptions {
  model?: EmbeddingModel;
  input: string | string[];
  /** Override the API key for this request only */
  apiKey?: string;
}

export interface EmbedResponse {
  /** Embedding vectors, one per input string */
  vectors: number[][];
  /** The actual embedding model used */
  model: string;
  /** Token usage */
  usage: { prompt_tokens: number; total_tokens: number };
}

// ─── Client Config ──────────────────────────────────────────────────────────

export interface ModelRouterConfig {
  /** Your Model Router API key (mr_sk_...) */
  apiKey: string;
  /** API base URL. Default: 'https://api.lxg2it.com/v1' */
  baseURL?: string;
  /** Default tier for all requests. Can be overridden per-request. */
  defaultTier?: Tier;
  /** Default prefer for all requests. Can be overridden per-request. */
  defaultPrefer?: Prefer;
  /** Request timeout in milliseconds. Default: 60000 */
  timeout?: number;
}

// ─── Error Types ────────────────────────────────────────────────────────────

export class ModelRouterError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
    public readonly code?: string,
  ) {
    super(message);
    this.name = 'ModelRouterError';
  }
}

export class MaxIterationsError extends ModelRouterError {
  constructor(
    public readonly iterations: number,
    public readonly partialMessages: Message[],
  ) {
    super(`Maximum iterations (${iterations}) reached before the model stopped calling tools`);
    this.name = 'MaxIterationsError';
  }
}
