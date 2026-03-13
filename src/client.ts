/**
 * @lxg2it/ai — ModelRouter client
 */

import OpenAI from 'openai';
import type {
  ModelRouterConfig,
  ChatOptions,
  ChatResponse,
  StreamOptions,
  StreamEvent,
  RunOptions,
  RunResponse,
  EmbedOptions,
  EmbedResponse,
  RoutingMetadata,
  Message,
  ToolCall,
  UsageInfo,
  ToolCallRecord,
  Tier,
  Provider,
  ToolDefinition,
} from './types.js';
import { MaxIterationsError } from './types.js';

const DEFAULT_BASE_URL = 'https://api.lxg2it.com/v1';

// ─── Header extraction ──────────────────────────────────────────────────────

/**
 * The OpenAI SDK doesn't expose HTTP response headers in the typed return value.
 * To get routing metadata, we provide a custom `fetch` implementation that captures
 * the X-Model-Router-* response headers into a closure-scoped store, then retrieve
 * them when building our wrapper response.
 */

type CapturedHeaders = {
  model: string | null;
  provider: string | null;
  tier: string | null;
  latency_ms: string | null;
};

function createHeaderCapturingFetch(): {
  fetch: typeof globalThis.fetch;
  getLastHeaders: () => CapturedHeaders;
} {
  let lastHeaders: CapturedHeaders = {
    model: null,
    provider: null,
    tier: null,
    latency_ms: null,
  };

  const wrappedFetch = async (input: Parameters<typeof globalThis.fetch>[0], init?: RequestInit): Promise<Response> => {
    const response = await globalThis.fetch(input as Parameters<typeof globalThis.fetch>[0], init);

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

function buildRoutingMetadata(
  headers: CapturedHeaders,
  defaultTier: Tier = 'standard',
): RoutingMetadata {
  return {
    model: headers.model ?? 'unknown',
    provider: (headers.provider ?? 'unknown') as Provider,
    tier: (headers.tier ?? defaultTier) as Tier,
    latency_ms: headers.latency_ms ? parseInt(headers.latency_ms, 10) : 0,
  };
}

// ─── ModelRouter class ─────────────────────────────────────────────────────

export class ModelRouter {
  private readonly config: Required<ModelRouterConfig>;
  private readonly headerCapture: ReturnType<typeof createHeaderCapturingFetch>;
  private readonly openai: OpenAI;

  constructor(config: ModelRouterConfig) {
    this.config = {
      baseURL: DEFAULT_BASE_URL,
      defaultTier: 'standard',
      defaultPrefer: 'balanced',
      timeout: 60_000,
      ...config,
    };

    this.headerCapture = createHeaderCapturingFetch();

    this.openai = new OpenAI({
      apiKey: config.apiKey,
      baseURL: this.config.baseURL,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      fetch: this.headerCapture.fetch as any,
      timeout: this.config.timeout,
    });
  }

  // ─── chat() ──────────────────────────────────────────────────────────────

  async chat(options: ChatOptions): Promise<ChatResponse> {
    const tier = options.tier ?? this.config.defaultTier;
    const prefer = options.prefer ?? this.config.defaultPrefer;

    const completion = await this.openai.chat.completions.create({
      model: tier,
      messages: options.messages as OpenAI.Chat.ChatCompletionMessageParam[],
      max_tokens: options.max_tokens,
      temperature: options.temperature,
      top_p: options.top_p,
      stop: options.stop,
      response_format: options.response_format,
      // Model router extensions passed as extra body params
      ...({ prefer, include_reasoning: options.include_reasoning } as Record<string, unknown>),
    } as OpenAI.Chat.ChatCompletionCreateParamsNonStreaming);

    const headers = this.headerCapture.getLastHeaders();
    const choice = completion.choices[0];
    const message = choice?.message;

    return {
      content: message?.content ?? '',
      reasoning:
        ((message as unknown as Record<string, unknown>)?.reasoning_content as string | null) ?? null,
      routing: buildRoutingMetadata(headers, tier),
      usage: completion.usage
        ? {
            prompt_tokens: completion.usage.prompt_tokens,
            completion_tokens: completion.usage.completion_tokens,
            total_tokens: completion.usage.total_tokens,
          }
        : null,
    };
  }

  // ─── stream() ────────────────────────────────────────────────────────────

  async *stream(options: StreamOptions): AsyncGenerator<StreamEvent> {
    const tier = options.tier ?? this.config.defaultTier;
    const prefer = options.prefer ?? this.config.defaultPrefer;

    try {
      const openaiStream = this.openai.beta.chat.completions.stream({
        model: tier,
        messages: options.messages as OpenAI.Chat.ChatCompletionMessageParam[],
        max_tokens: options.max_tokens,
        temperature: options.temperature,
        top_p: options.top_p,
        stop: options.stop,
        stream: true,
        ...({ prefer, include_reasoning: options.include_reasoning } as Record<string, unknown>),
      } as OpenAI.Chat.ChatCompletionCreateParamsStreaming);

      // Accumulate tool call argument deltas
      const toolCallAccumulators = new Map<
        number,
        { id: string; name: string; argumentsBuffer: string }
      >();

      for await (const chunk of openaiStream) {
        const delta = chunk.choices[0]?.delta;
        if (!delta) continue;

        // Reasoning content delta
        const reasoningDelta = (delta as Record<string, unknown>).reasoning_content as
          | string
          | undefined;
        if (reasoningDelta) {
          yield { type: 'reasoning', delta: reasoningDelta };
        }

        // Content delta
        if (delta.content) {
          yield { type: 'content', delta: delta.content };
        }

        // Tool call deltas — accumulate until complete
        if (delta.tool_calls) {
          for (const toolCallDelta of delta.tool_calls) {
            const index = toolCallDelta.index;

            if (toolCallDelta.id) {
              // New tool call starting
              toolCallAccumulators.set(index, {
                id: toolCallDelta.id,
                name: toolCallDelta.function?.name ?? '',
                argumentsBuffer: toolCallDelta.function?.arguments ?? '',
              });
            } else {
              // Argument delta for existing tool call
              const acc = toolCallAccumulators.get(index);
              if (acc) {
                acc.argumentsBuffer += toolCallDelta.function?.arguments ?? '';
                yield {
                  type: 'tool_call_delta',
                  id: acc.id,
                  name: acc.name,
                  argumentsDelta: toolCallDelta.function?.arguments ?? '',
                };
              }
            }
          }
        }

        // Finish reason — emit complete tool calls
        const finishReason = chunk.choices[0]?.finish_reason;
        if (finishReason === 'tool_calls') {
          for (const acc of toolCallAccumulators.values()) {
            yield {
              type: 'tool_call',
              id: acc.id,
              name: acc.name,
              arguments: acc.argumentsBuffer,
            };
          }
          toolCallAccumulators.clear();
        }
      }

      // Stream ended — emit done with routing metadata
      const headers = this.headerCapture.getLastHeaders();
      const finalCompletion = await openaiStream.finalChatCompletion();

      yield {
        type: 'done',
        routing: buildRoutingMetadata(headers, tier),
        usage: finalCompletion.usage
          ? {
              prompt_tokens: finalCompletion.usage.prompt_tokens,
              completion_tokens: finalCompletion.usage.completion_tokens,
              total_tokens: finalCompletion.usage.total_tokens,
            }
          : null,
      };
    } catch (error) {
      yield {
        type: 'error',
        error: error instanceof Error ? error : new Error(String(error)),
      };
    }
  }

  // ─── run() ────────────────────────────────────────────────────────────────

  async run(options: RunOptions): Promise<RunResponse> {
    const { tools, maxIterations = 10, onMaxIterations = 'error', ...chatOptions } = options;

    const toolsSpec = tools.map((t) => ({
      type: 'function' as const,
      function: {
        name: t.name,
        description: t.description,
        parameters: t.jsonSchema,
      },
    }));

    const toolsByName = new Map(tools.map((t) => [t.name, t]));

    let messages: Message[] = [...options.messages];
    const allToolCalls: ToolCallRecord[] = [];
    let aggregateUsage: UsageInfo = { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
    let lastRouting: RoutingMetadata | null = null;
    let iterations = 0;

    while (iterations < maxIterations) {
      iterations++;

      const tier = chatOptions.tier ?? this.config.defaultTier;
      const prefer = chatOptions.prefer ?? this.config.defaultPrefer;

      const completion = await this.openai.chat.completions.create({
        model: tier,
        messages: messages as OpenAI.Chat.ChatCompletionMessageParam[],
        tools: toolsSpec,
        tool_choice: 'auto',
        max_tokens: chatOptions.max_tokens,
        temperature: chatOptions.temperature,
        ...({ prefer } as Record<string, unknown>),
      } as OpenAI.Chat.ChatCompletionCreateParamsNonStreaming);

      const headers = this.headerCapture.getLastHeaders();
      lastRouting = buildRoutingMetadata(headers, tier);

      if (completion.usage) {
        aggregateUsage.prompt_tokens += completion.usage.prompt_tokens;
        aggregateUsage.completion_tokens += completion.usage.completion_tokens;
        aggregateUsage.total_tokens += completion.usage.total_tokens;
      }

      const choice = completion.choices[0];
      const assistantMessage = choice?.message;

      if (!assistantMessage) break;

      messages.push({
        role: 'assistant',
        content: assistantMessage.content ?? '',
        tool_calls: assistantMessage.tool_calls as ToolCall[] | undefined,
      });

      // If no more tool calls, we're done
      if (choice.finish_reason !== 'tool_calls' || !assistantMessage.tool_calls?.length) {
        return {
          content: assistantMessage.content ?? '',
          toolCalls: allToolCalls,
          iterations,
          routing: lastRouting,
          usage: aggregateUsage,
          messages,
        };
      }

      // Execute all tool calls in parallel
      const toolResults = await Promise.all(
        assistantMessage.tool_calls.map(async (toolCall) => {
          const toolDef = toolsByName.get(toolCall.function.name);
          if (!toolDef) {
            return {
              id: toolCall.id,
              name: toolCall.function.name,
              args: {},
              result: { error: `Tool "${toolCall.function.name}" not found` },
            };
          }

          try {
            const args = toolDef.parameters.parse(JSON.parse(toolCall.function.arguments));
            const result = await toolDef.execute(args);
            return { id: toolCall.id, name: toolCall.function.name, args, result };
          } catch (err) {
            return {
              id: toolCall.id,
              name: toolCall.function.name,
              args: {},
              result: { error: String(err) },
            };
          }
        }),
      );

      for (const tr of toolResults) {
        allToolCalls.push({
          name: tr.name,
          args: tr.args as Record<string, unknown>,
          result: tr.result,
        });
        messages.push({
          role: 'tool',
          tool_call_id: tr.id,
          content: JSON.stringify(tr.result),
        });
      }
    }

    // Reached maxIterations
    if (onMaxIterations === 'error') {
      throw new MaxIterationsError(iterations, messages);
    }

    // Return partial result
    const lastAssistant = [...messages].reverse().find((m) => m.role === 'assistant');
    return {
      content: lastAssistant?.role === 'assistant' ? lastAssistant.content : '',
      toolCalls: allToolCalls,
      iterations,
      routing: lastRouting!,
      usage: aggregateUsage,
      messages,
    };
  }

  // ─── embed() ─────────────────────────────────────────────────────────────

  async embed(options: EmbedOptions): Promise<EmbedResponse> {
    const model = options.model ?? 'embed-small';

    const response = await this.openai.embeddings.create({
      model,
      input: options.input,
    });

    return {
      vectors: response.data.map((d) => d.embedding),
      model: response.model,
      usage: {
        prompt_tokens: response.usage.prompt_tokens,
        total_tokens: response.usage.total_tokens,
      },
    };
  }
}

// ─── tool() factory ────────────────────────────────────────────────────────

interface ToolOptions<TParams, TResult> {
  name: string;
  description?: string;
  parameters: {
    parse: (input: unknown) => TParams;
    _def?: unknown;
  };
  execute: (params: TParams) => Promise<TResult>;
}

/**
 * Create a typed tool definition for use with ai.run().
 *
 * Uses Zod schemas for parameter validation and TypeScript type inference.
 * Zod is an optional peer dependency — any object with a compatible `parse()`
 * method works for validation, though JSON schema generation requires Zod.
 *
 * For JSON schema generation from Zod schemas, uses `zod-to-json-schema`.
 *
 * @example
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
export function tool<TParams, TResult>(
  options: ToolOptions<TParams, TResult>,
): ToolDefinition<TParams, TResult> {
  let jsonSchema: Record<string, unknown> = { type: 'object', properties: {} };

  if (options.parameters._def) {
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const { zodToJsonSchema } = require('zod-to-json-schema');
      jsonSchema = zodToJsonSchema(options.parameters, { target: 'openAi' }) as Record<
        string,
        unknown
      >;
    } catch {
      // zod-to-json-schema not available or not a Zod schema — use empty schema
    }
  }

  return {
    name: options.name,
    description: options.description,
    parameters: options.parameters,
    execute: options.execute,
    jsonSchema,
  };
}
