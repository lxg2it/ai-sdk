/**
 * @lxg2it/ai — Public API entry point
 */

export { ModelRouter, tool } from './client.js';
export type {
  // Config
  ModelRouterConfig,
  // Tiers and preferences
  Tier,
  Prefer,
  EmbeddingModel,
  Provider,
  // Messages
  Message,
  SystemMessage,
  UserMessage,
  AssistantMessage,
  ToolResultMessage,
  ContentPart,
  ToolCall,
  // Routing
  RoutingMetadata,
  UsageInfo,
  // chat()
  ChatOptions,
  ChatResponse,
  // stream()
  StreamOptions,
  StreamEvent,
  // tool()
  ToolDefinition,
  // run()
  RunOptions,
  RunResponse,
  ToolCallRecord,
  // embed()
  EmbedOptions,
  EmbedResponse,
} from './types.js';
export { ModelRouterError, MaxIterationsError } from './types.js';
