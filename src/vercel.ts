/**
 * @lxg2it/ai/vercel — Vercel AI SDK provider adapter
 *
 * Exposes @lxg2it/ai as a Vercel AI SDK compatible provider.
 * Requires `ai` package as peer dependency (v4+).
 *
 * @example
 * ```typescript
 * import { createModelRouter } from '@lxg2it/ai/vercel';
 * import { streamText } from 'ai';
 *
 * const mr = createModelRouter({ apiKey: 'mr_sk_...' });
 *
 * // Simple usage
 * const result = await streamText({
 *   model: mr('standard'),
 *   messages: [{ role: 'user', content: 'Hello' }],
 * });
 *
 * // With preferences
 * const fastResult = await streamText({
 *   model: mr('standard', { prefer: 'fast' }),
 *   messages: [{ role: 'user', content: 'Quick question' }],
 * });
 *
 * // Access routing metadata
 * for await (const chunk of result.fullStream) {
 *   if (chunk.type === 'finish') {
 *     const meta = chunk.providerMetadata?.modelRouter;
 *     console.log(`Used: ${meta?.model} (${meta?.provider})`);
 *   }
 * }
 * ```
 */

import OpenAI from 'openai';
import type { ModelRouterConfig, Tier, Prefer, Provider } from './types.js';

// ─── Types from @ai-sdk/provider ────────────────────────────────────────────
//
// We inline a minimal subset of the LanguageModelV1 interface rather than
// importing it, to avoid making `ai` a hard dependency of this file.

interface LanguageModelV1 {
  readonly specificationVersion: 'v1';
  readonly provider: string;
  readonly modelId: string;
  readonly defaultObjectGenerationMode: 'json' | 'tool' | undefined;
  doGenerate(options: LanguageModelV1CallOptions): Promise<LanguageModelV1GenerateResult>;
  doStream(options: LanguageModelV1CallOptions): Promise<{
    stream: ReadableStream<LanguageModelV1StreamPart>;
    rawCall: { rawPrompt: unknown; rawSettings: Record<string, unknown> };
    rawResponse?: { headers?: Record<string, string> };
  }>;
}

interface LanguageModelV1CallOptions {
  mode:
    | { type: 'regular'; tools?: unknown[] }
    | { type: 'object-json' }
    | { type: 'object-tool' };
  prompt: LanguageModelV1Message[];
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  presencePenalty?: number;
  frequencyPenalty?: number;
  seed?: number;
  headers?: Record<string, string | undefined>;
  abortSignal?: AbortSignal;
}

type LanguageModelV1Message =
  | { role: 'system'; content: string }
  | {
      role: 'user';
      content: Array<
        | { type: 'text'; text: string }
        | { type: 'image'; image: URL | Uint8Array }
      >;
    }
  | {
      role: 'assistant';
      content: Array<
        | { type: 'text'; text: string }
        | { type: 'tool-call'; toolCallId: string; toolName: string; args: unknown }
      >;
    }
  | {
      role: 'tool';
      content: Array<{
        type: 'tool-result';
        toolCallId: string;
        toolName: string;
        result: unknown;
      }>;
    };

interface LanguageModelV1GenerateResult {
  text?: string;
  toolCalls?: Array<{
    toolCallType: 'function';
    toolCallId: string;
    toolName: string;
    args: string;
  }>;
  finishReason:
    | 'stop'
    | 'length'
    | 'tool-calls'
    | 'content-filter'
    | 'error'
    | 'other'
    | 'unknown';
  usage: { promptTokens: number; completionTokens: number };
  rawCall: { rawPrompt: unknown; rawSettings: Record<string, unknown> };
  rawResponse?: { headers?: Record<string, string> };
  providerMetadata?: {
    modelRouter?: {
      model: string;
      provider: Provider;
      tier: string;
      latency_ms: number;
    };
  };
}

type LanguageModelV1FinishReason =
  | 'stop'
  | 'length'
  | 'tool-calls'
  | 'content-filter'
  | 'error'
  | 'other'
  | 'unknown';

type LanguageModelV1StreamPart =
  | { type: 'text-delta'; textDelta: string }
  | {
      type: 'tool-call-delta';
      toolCallType: 'function';
      toolCallId: string;
      toolName: string;
      argsTextDelta: string;
    }
  | {
      type: 'tool-call';
      toolCallType: 'function';
      toolCallId: string;
      toolName: string;
      args: string;
    }
  | {
      type: 'finish';
      finishReason: LanguageModelV1FinishReason;
      usage: { promptTokens: number; completionTokens: number };
      providerMetadata?: Record<string, unknown>;
    }
  | { type: 'error'; error: unknown };

// ─── Prompt conversion ────────────────────────────────────────────────────

function convertPromptToOpenAI(
  prompt: LanguageModelV1Message[],
): OpenAI.Chat.ChatCompletionMessageParam[] {
  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [];

  for (const message of prompt) {
    switch (message.role) {
      case 'system':
        messages.push({ role: 'system', content: message.content });
        break;

      case 'user': {
        const imageParts = message.content.filter((p) => p.type === 'image');

        if (imageParts.length === 0) {
          messages.push({
            role: 'user',
            content: message.content
              .filter((p): p is { type: 'text'; text: string } => p.type === 'text')
              .map((p) => p.text)
              .join(''),
          });
        } else {
          messages.push({
            role: 'user',
            content: message.content.map((part) => {
              if (part.type === 'text') {
                return { type: 'text' as const, text: part.text };
              } else {
                const img = (part as { type: 'image'; image: URL | Uint8Array }).image;
                const url =
                  img instanceof URL
                    ? img.toString()
                    : `data:image/jpeg;base64,${Buffer.from(img).toString('base64')}`;
                return { type: 'image_url' as const, image_url: { url } };
              }
            }),
          });
        }
        break;
      }

      case 'assistant': {
        const textContent = message.content
          .filter((p): p is { type: 'text'; text: string } => p.type === 'text')
          .map((p) => p.text)
          .join('');

        const toolCallParts = message.content.filter(
          (p): p is { type: 'tool-call'; toolCallId: string; toolName: string; args: unknown } =>
            p.type === 'tool-call',
        );

        if (toolCallParts.length > 0) {
          messages.push({
            role: 'assistant',
            content: textContent || null,
            tool_calls: toolCallParts.map((tc) => ({
              id: tc.toolCallId,
              type: 'function' as const,
              function: {
                name: tc.toolName,
                arguments: JSON.stringify(tc.args),
              },
            })),
          });
        } else {
          messages.push({ role: 'assistant', content: textContent });
        }
        break;
      }

      case 'tool':
        for (const part of message.content) {
          messages.push({
            role: 'tool',
            tool_call_id: part.toolCallId,
            content: JSON.stringify(part.result),
          });
        }
        break;
    }
  }

  return messages;
}

// ─── Header capturing fetch ──────────────────────────────────────────────

function createHeaderCapturingFetch(): {
  fetch: typeof globalThis.fetch;
  getLastHeaders: () => Record<string, string | null>;
} {
  let lastHeaders: Record<string, string | null> = {};

  const wrappedFetch = async (input: Parameters<typeof globalThis.fetch>[0], init?: RequestInit): Promise<Response> => {
    const response = await globalThis.fetch(input as Parameters<typeof globalThis.fetch>[0], init);
    lastHeaders = {
      'x-model-router-model': response.headers.get('X-Model-Router-Model'),
      'x-model-router-provider': response.headers.get('X-Model-Router-Provider'),
      'x-model-router-tier': response.headers.get('X-Model-Router-Tier'),
      'x-model-router-latency-ms': response.headers.get('X-Model-Router-Latency-Ms'),
    };
    return response;
  };

  return { fetch: wrappedFetch, getLastHeaders: () => ({ ...lastHeaders }) };
}

// ─── ModelRouterLanguageModel ────────────────────────────────────────────

const FINISH_REASON_MAP: Record<string, LanguageModelV1FinishReason> = {
  stop: 'stop',
  length: 'length',
  tool_calls: 'tool-calls',
  content_filter: 'content-filter',
};

class ModelRouterLanguageModel implements LanguageModelV1 {
  readonly specificationVersion = 'v1' as const;
  readonly provider = 'model-router';
  readonly modelId: string;
  readonly defaultObjectGenerationMode = 'json' as const;

  private readonly tier: Tier;
  private readonly prefer: Prefer | undefined;
  private readonly openai: OpenAI;
  private readonly headerCapture: ReturnType<typeof createHeaderCapturingFetch>;

  constructor(
    tier: Tier,
    options: { prefer?: Prefer; apiKey: string; baseURL?: string } = { apiKey: '' },
  ) {
    this.tier = tier;
    this.prefer = options.prefer;
    this.modelId = options.prefer ? `${tier}:${options.prefer}` : tier;

    this.headerCapture = createHeaderCapturingFetch();
    this.openai = new OpenAI({
      apiKey: options.apiKey,
      baseURL: options.baseURL ?? 'https://api.lxg2it.com/v1',
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      fetch: this.headerCapture.fetch as any,
    });
  }

  async doGenerate(options: LanguageModelV1CallOptions): Promise<LanguageModelV1GenerateResult> {
    const messages = convertPromptToOpenAI(options.prompt);

    const tools =
      options.mode.type === 'regular' && options.mode.tools
        ? (options.mode.tools as OpenAI.Chat.ChatCompletionTool[])
        : undefined;

    const completion = await this.openai.chat.completions.create({
      model: this.tier,
      messages,
      tools,
      tool_choice: tools ? 'auto' : undefined,
      max_tokens: options.maxTokens,
      temperature: options.temperature,
      top_p: options.topP,
      ...({ prefer: this.prefer } as Record<string, unknown>),
    } as OpenAI.Chat.ChatCompletionCreateParamsNonStreaming);

    const headers = this.headerCapture.getLastHeaders();
    const choice = completion.choices[0];
    const message = choice?.message;

    return {
      text: message?.content ?? undefined,
      toolCalls: message?.tool_calls?.map((tc) => ({
        toolCallType: 'function' as const,
        toolCallId: tc.id,
        toolName: tc.function.name,
        args: tc.function.arguments,
      })),
      finishReason: FINISH_REASON_MAP[choice?.finish_reason ?? 'stop'] ?? 'unknown',
      usage: {
        promptTokens: completion.usage?.prompt_tokens ?? 0,
        completionTokens: completion.usage?.completion_tokens ?? 0,
      },
      rawCall: { rawPrompt: messages, rawSettings: {} },
      rawResponse: {
        headers: Object.fromEntries(
          Object.entries(headers).filter((entry): entry is [string, string] => entry[1] !== null),
        ),
      },
      providerMetadata: {
        modelRouter: {
          model: headers['x-model-router-model'] ?? 'unknown',
          provider: (headers['x-model-router-provider'] ?? 'unknown') as Provider,
          tier: headers['x-model-router-tier'] ?? this.tier,
          latency_ms: parseInt(headers['x-model-router-latency-ms'] ?? '0', 10),
        },
      },
    };
  }

  async doStream(options: LanguageModelV1CallOptions): Promise<{
    stream: ReadableStream<LanguageModelV1StreamPart>;
    rawCall: { rawPrompt: unknown; rawSettings: Record<string, unknown> };
    rawResponse?: { headers?: Record<string, string> };
  }> {
    const messages = convertPromptToOpenAI(options.prompt);

    const tools =
      options.mode.type === 'regular' && options.mode.tools
        ? (options.mode.tools as OpenAI.Chat.ChatCompletionTool[])
        : undefined;

    const openaiStream = await this.openai.chat.completions.create({
      model: this.tier,
      messages,
      tools,
      tool_choice: tools ? 'auto' : undefined,
      max_tokens: options.maxTokens,
      temperature: options.temperature,
      top_p: options.topP,
      stream: true,
      stream_options: { include_usage: true },
      ...({ prefer: this.prefer } as Record<string, unknown>),
    } as OpenAI.Chat.ChatCompletionCreateParamsStreaming);

    const headerCapture = this.headerCapture;
    const tier = this.tier;

    const stream = new ReadableStream<LanguageModelV1StreamPart>({
      async start(controller) {
        try {
          let usage = { promptTokens: 0, completionTokens: 0 };
          const toolCallAccumulators = new Map<
            number,
            { id: string; name: string; args: string }
          >();

          for await (const chunk of openaiStream) {
            const delta = chunk.choices[0]?.delta;

            if (delta?.content) {
              controller.enqueue({ type: 'text-delta', textDelta: delta.content });
            }

            if (delta?.tool_calls) {
              for (const tcDelta of delta.tool_calls) {
                const idx = tcDelta.index;
                if (tcDelta.id) {
                  toolCallAccumulators.set(idx, {
                    id: tcDelta.id,
                    name: tcDelta.function?.name ?? '',
                    args: '',
                  });
                  controller.enqueue({
                    type: 'tool-call-delta',
                    toolCallType: 'function',
                    toolCallId: tcDelta.id,
                    toolName: tcDelta.function?.name ?? '',
                    argsTextDelta: tcDelta.function?.arguments ?? '',
                  });
                } else {
                  const acc = toolCallAccumulators.get(idx);
                  if (acc) {
                    acc.args += tcDelta.function?.arguments ?? '';
                    controller.enqueue({
                      type: 'tool-call-delta',
                      toolCallType: 'function',
                      toolCallId: acc.id,
                      toolName: acc.name,
                      argsTextDelta: tcDelta.function?.arguments ?? '',
                    });
                  }
                }
              }
            }

            if (chunk.usage) {
              usage = {
                promptTokens: chunk.usage.prompt_tokens,
                completionTokens: chunk.usage.completion_tokens,
              };
            }

            const finishReason = chunk.choices[0]?.finish_reason;
            if (finishReason === 'tool_calls') {
              for (const acc of toolCallAccumulators.values()) {
                controller.enqueue({
                  type: 'tool-call',
                  toolCallType: 'function',
                  toolCallId: acc.id,
                  toolName: acc.name,
                  args: acc.args,
                });
              }
            }

            if (finishReason) {
              const headers = headerCapture.getLastHeaders();
              controller.enqueue({
                type: 'finish',
                finishReason:
                  (FINISH_REASON_MAP[finishReason] as LanguageModelV1FinishReason) ?? 'unknown',
                usage,
                providerMetadata: {
                  modelRouter: {
                    model: headers['x-model-router-model'] ?? 'unknown',
                    provider: headers['x-model-router-provider'] ?? 'unknown',
                    tier: headers['x-model-router-tier'] ?? tier,
                    latency_ms: parseInt(headers['x-model-router-latency-ms'] ?? '0', 10),
                  },
                },
              });
            }
          }

          controller.close();
        } catch (err) {
          controller.enqueue({ type: 'error', error: err });
          controller.close();
        }
      },
    });

    return {
      stream,
      rawCall: { rawPrompt: messages, rawSettings: {} },
    };
  }
}

// ─── createModelRouter factory ─────────────────────────────────────────────

type ModelRouterVercelConfig = Pick<ModelRouterConfig, 'apiKey' | 'baseURL'>;

/**
 * Create a Vercel AI SDK provider for Model Router.
 *
 * Returns a factory function that creates LanguageModelV1 instances.
 */
export function createModelRouter(
  config: ModelRouterVercelConfig,
): (tier: Tier, options?: { prefer?: Prefer }) => LanguageModelV1 {
  return function modelFactory(
    tier: Tier,
    options?: { prefer?: Prefer },
  ): LanguageModelV1 {
    return new ModelRouterLanguageModel(tier, {
      apiKey: config.apiKey,
      baseURL: config.baseURL,
      prefer: options?.prefer,
    });
  };
}
