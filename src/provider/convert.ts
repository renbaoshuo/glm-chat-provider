import * as vscode from 'vscode';
import secureJsonParse from 'secure-json-parse';
import {P, match} from 'ts-pattern';
import type {GlmMessage, GlmTool, GlmToolCall} from '../api';
import {readThinkingText} from './thinking';

export type ToolCallBuilder = {
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

export function parseToolArguments(argumentsText: string): Record<string, unknown> {
  const parsed = secureJsonParse.safeParse(argumentsText || '{}');
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return {};
  }
  return parsed as Record<string, unknown>;
}

export function convertMessages(
  messages: readonly vscode.LanguageModelChatRequestMessage[],
): GlmMessage[] {
  return messages.map(message => toGlmMessage(message));
}

function toGlmMessage(
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
            content: value.content
              .map(item =>
                item instanceof vscode.LanguageModelTextPart ? item.value : '',
              )
              .join(''),
          },
        }))
        .otherwise(value => {
          const thinking = readThinkingText(value);
          return thinking ? {...state, text: state.text + thinking} : state;
        }),
    {text: '', toolCalls: []},
  );

  const role = mapRole(message.role);

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

  return {role, content: accumulated.text};
}

function mapRole(
  role: vscode.LanguageModelChatMessageRole,
): 'user' | 'assistant' | 'system' {
  return match(role)
    .with(vscode.LanguageModelChatMessageRole.Assistant, () => 'assistant' as const)
    .with(vscode.LanguageModelChatMessageRole.User, () => 'user' as const)
    .otherwise(() => 'system' as const);
}

export function convertTools(
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
