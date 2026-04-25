import * as vscode from 'vscode';
import secureJsonParse from 'secure-json-parse';
import {P, match} from 'ts-pattern';
import {
  GlmApiClient,
  GlmApiError,
  type GlmMessage,
  type GlmTool,
  type GlmToolCall,
} from './api';
import type {ChatCompletionChunk} from 'openai/resources/chat/completions/completions';
import type {AuthManager} from './auth';
import {GLM_MODELS} from './models';
export {GLM_MODELS};

type ToolCallBuilder = {
  id: string;
  name: string;
  arguments: string;
};

type ToolResult = {
  callId: string;
  content: string;
};

type MessageAccumulator = {
  text: string;
  toolCalls: GlmToolCall[];
  toolResult?: ToolResult;
};

type ThinkingState = {
  buffer: string;
  insideThinking: boolean;
};

type ModelWithApiKey = vscode.LanguageModelChatInformation & {
  __glmApiKey?: string;
};

type PrepareLanguageModelChatInfoOptions =
  vscode.PrepareLanguageModelChatModelOptions & {
    readonly configuration?: {
      readonly apiKey?: string;
      readonly [key: string]: unknown;
    };
  };

const THINK_OPEN = '<think>';
const THINK_CLOSE = '</think>';
const THINK_OPEN_MARKDOWN = '<details><summary>Thinking</summary>\n\n';
const THINK_CLOSE_MARKDOWN = '\n\n</details>\n\n';

const TYPED_MODELS: vscode.LanguageModelChatInformation[] = GLM_MODELS.map(
  m => ({...m}),
);

function parseToolArguments(argumentsText: string): Record<string, unknown> {
  const parsed = secureJsonParse.safeParse(argumentsText || '{}');
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return {};
  }
  return parsed as Record<string, unknown>;
}

function appendThinkingSegment(
  segment: string,
  insideThinking: boolean,
): string {
  return insideThinking
    ? `${segment}${THINK_CLOSE_MARKDOWN}`
    : `${segment}${THINK_OPEN_MARKDOWN}`;
}

function processThinkingContent(
  content: string,
  state: ThinkingState,
): {output: string; state: ThinkingState} {
  let output = '';
  const buffer = state.buffer ? state.buffer + content : content;
  const insideThinking = state.insideThinking;

  while (true) {
    const marker = insideThinking ? THINK_CLOSE : THINK_OPEN;
    const markerIndex = buffer.indexOf(marker);

    if (markerIndex >= 0) {
      output += appendThinkingSegment(
        buffer.slice(0, markerIndex),
        insideThinking,
      );
      const remaining = buffer.slice(markerIndex + marker.length);
      if (!remaining) {
        return {output, state: {buffer: '', insideThinking: !insideThinking}};
      }
      return processThinkingContent(remaining, {
        buffer: output,
        insideThinking: !insideThinking,
      });
    }

    const maxKeep = Math.min(buffer.length, marker.length - 1);
    let keep = 0;
    for (let i = maxKeep; i > 0; i--) {
      if (marker.startsWith(buffer.slice(buffer.length - i))) {
        keep = i;
        break;
      }
    }

    output += buffer.slice(0, buffer.length - keep);
    return {
      output,
      state: {
        buffer: buffer.slice(buffer.length - keep) || '',
        insideThinking,
      },
    };
  }
}

export class GlmChatProvider implements vscode.LanguageModelChatProvider {
  private readonly _onDidChangeLanguageModelChatInformation =
    new vscode.EventEmitter<void>();

  readonly onDidChangeLanguageModelChatInformation =
    this._onDidChangeLanguageModelChatInformation.event;

  constructor(private readonly authManager: AuthManager) {}

  fireLanguageModelChatInformationChange(): void {
    this._onDidChangeLanguageModelChatInformation.fire();
  }

  async provideLanguageModelChatInformation(
    options: PrepareLanguageModelChatInfoOptions,
    token: vscode.CancellationToken,
  ): Promise<vscode.LanguageModelChatInformation[]> {
    void token;
    if (options.configuration === undefined) {
      return [];
    }

    const raw = options.configuration.apiKey;
    const apiKey =
      typeof raw === 'string' && raw.trim().length > 0 ? raw.trim() : undefined;

    if (!apiKey) {
      return [];
    }

    return this.modelsWithApiKey(apiKey);
  }

  private modelsWithApiKey(
    apiKey: string,
  ): vscode.LanguageModelChatInformation[] {
    return TYPED_MODELS.map(model => ({
      ...model,
      __glmApiKey: apiKey,
    }));
  }

  async provideLanguageModelChatResponse(
    model: vscode.LanguageModelChatInformation,
    messages: readonly vscode.LanguageModelChatRequestMessage[],
    options: vscode.ProvideLanguageModelChatResponseOptions,
    progress: vscode.Progress<vscode.LanguageModelResponsePart>,
    token: vscode.CancellationToken,
  ): Promise<void> {
    const modelApiKey = (model as ModelWithApiKey).__glmApiKey;
    const apiKey =
      modelApiKey && modelApiKey.trim().length > 0
        ? modelApiKey
        : await this.authManager.getOrPromptApiKey();

    if (!apiKey) {
      throw new Error(
        'API key not configured. Use "GLM: Set API Key" command.',
      );
    }

    try {
      await this.streamResponse(
        new GlmApiClient(apiKey),
        model,
        messages,
        options,
        progress,
        token,
      );
    } catch (error) {
      await this.throwMappedError(error);
    }
  }

  private resolveThinking(): {type: 'enabled' | 'disabled'} | undefined {
    const config = vscode.workspace
      .getConfiguration('glm-chat-provider')
      .get<string>('defaultThinkingMode', 'auto');
    if (config === 'enabled') {
      return {type: 'enabled'};
    }
    if (config === 'disabled') {
      return {type: 'disabled'};
    }
    return undefined;
  }

  private async streamResponse(
    client: GlmApiClient,
    model: vscode.LanguageModelChatInformation,
    messages: readonly vscode.LanguageModelChatRequestMessage[],
    options: vscode.ProvideLanguageModelChatResponseOptions,
    progress: vscode.Progress<vscode.LanguageModelResponsePart>,
    token: vscode.CancellationToken,
  ): Promise<void> {
    const toolCallBuilders = new Map<number, ToolCallBuilder>();
    let thinkingState: ThinkingState = {buffer: '', insideThinking: false};

    const stream = client.streamChat(
      model.id,
      this.convertMessages(messages),
      {
        maxTokens: options.modelOptions?.maxTokens as number | undefined,
        tools: this.convertTools(options.tools),
        thinking: this.resolveThinking(),
      },
      token,
    );

    for await (const chunk of stream) {
      if (token.isCancellationRequested) {
        return;
      }

      for (const choice of chunk.choices) {
        thinkingState = this.reportTextDelta(
          choice.delta.content,
          thinkingState,
          progress,
        );
        this.collectToolCalls(choice.delta.tool_calls, toolCallBuilders);
        if (choice.finish_reason === 'tool_calls') {
          this.reportToolCalls(progress, toolCallBuilders);
        }
      }
    }

    this.reportToolCalls(progress, toolCallBuilders);
  }

  private reportTextDelta(
    content: string | null | undefined,
    state: ThinkingState,
    progress: vscode.Progress<vscode.LanguageModelResponsePart>,
  ): ThinkingState {
    if (!content) {
      return state;
    }

    const result = processThinkingContent(content, state);
    if (result.output) {
      progress.report(new vscode.LanguageModelTextPart(result.output));
    }
    return result.state;
  }

  private collectToolCalls(
    toolCalls: ChatCompletionChunk.Choice.Delta.ToolCall[] | undefined,
    builders: Map<number, ToolCallBuilder>,
  ): void {
    if (!toolCalls?.length) {
      return;
    }

    for (const call of toolCalls) {
      const builder = builders.get(call.index) ?? {
        id: '',
        name: '',
        arguments: '',
      };

      if (call.id) {
        builder.id = call.id;
      }
      if (call.function?.name) {
        builder.name = call.function.name;
      }
      if (call.function?.arguments) {
        builder.arguments += call.function.arguments;
      }

      builders.set(call.index, builder);
    }
  }

  private reportToolCalls(
    progress: vscode.Progress<vscode.LanguageModelResponsePart>,
    builders: Map<number, ToolCallBuilder>,
  ): void {
    if (builders.size === 0) {
      return;
    }

    for (const builder of builders.values()) {
      if (!builder.id || !builder.name) {
        continue;
      }

      progress.report(
        new vscode.LanguageModelToolCallPart(
          builder.id,
          builder.name,
          parseToolArguments(builder.arguments),
        ),
      );
    }

    builders.clear();
  }

  private async throwMappedError(error: unknown): Promise<never> {
    if (!(error instanceof GlmApiError)) {
      throw error;
    }

    await match(error.statusCode)
      .with(401, async () => {
        await this.authManager.deleteApiKey();
        throw new Error(
          'Invalid API key. Please set a new one using "GLM: Set API Key".',
        );
      })
      .with(429, async () => {
        throw new Error('Rate limit exceeded. Please wait and try again.');
      })
      .otherwise(async () => {
        throw new Error(`GLM API error: ${error.message}`);
      });

    throw error;
  }

  provideTokenCount(
    model: vscode.LanguageModelChatInformation,
    text: string | vscode.LanguageModelChatRequestMessage,
    token: vscode.CancellationToken,
  ): Thenable<number> {
    void model;
    void token;
    if (typeof text === 'string') {
      return Promise.resolve(Math.ceil(text.length / 4));
    }

    let totalChars = 0;
    for (const part of text.content) {
      if (part instanceof vscode.LanguageModelTextPart) {
        totalChars += part.value.length;
      }
    }
    return Promise.resolve(Math.ceil(totalChars / 4));
  }

  private convertMessages(
    messages: readonly vscode.LanguageModelChatRequestMessage[],
  ): GlmMessage[] {
    return messages.map(message => this.toGlmMessage(message));
  }

  private toGlmMessage(
    message: vscode.LanguageModelChatRequestMessage,
  ): GlmMessage {
    const accumulated = message.content.reduce<MessageAccumulator>(
      (state, part) =>
        match(part)
          .with(P.instanceOf(vscode.LanguageModelTextPart), value => ({
            ...state,
            text: state.text + value.value,
          }))
          .with(P.instanceOf(vscode.LanguageModelToolCallPart), value => ({
            ...state,
            toolCalls: [
              ...state.toolCalls,
              {
                id: value.callId,
                type: 'function' as const,
                function: {
                  name: value.name,
                  arguments: JSON.stringify(value.input),
                },
              },
            ],
          }))
          .with(P.instanceOf(vscode.LanguageModelToolResultPart), value => ({
            ...state,
            toolResult: {
              callId: value.callId,
              content:
                typeof value.content === 'string'
                  ? value.content
                  : JSON.stringify(value.content),
            },
          }))
          .otherwise(() => state),
      {
        text: '',
        toolCalls: [],
      },
    );

    if (accumulated.toolResult) {
      return {
        role: 'tool',
        content: accumulated.toolResult.content,
        tool_call_id: accumulated.toolResult.callId,
      };
    }

    if (accumulated.toolCalls.length > 0) {
      return {
        role: 'assistant',
        content: accumulated.text,
        tool_calls: accumulated.toolCalls,
      };
    }

    return {
      role: this.convertRole(message.role),
      content: accumulated.text,
      name: message.name,
    };
  }

  private convertRole(
    role: vscode.LanguageModelChatMessageRole,
  ): 'system' | 'user' | 'assistant' {
    return match(role)
      .with(
        vscode.LanguageModelChatMessageRole.Assistant,
        () => 'assistant' as const,
      )
      .with(vscode.LanguageModelChatMessageRole.User, () => 'user' as const)
      .otherwise(() => 'system' as const);
  }

  private convertTools(
    tools?: readonly vscode.LanguageModelChatTool[],
  ): GlmTool[] | undefined {
    return tools?.length
      ? tools.map(tool => ({
          type: 'function',
          function: {
            name: tool.name,
            description: tool.description,
            parameters: (tool.inputSchema ?? {}) as Record<string, unknown>,
          },
        }))
      : undefined;
  }
}
