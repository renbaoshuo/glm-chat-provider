import * as vscode from 'vscode';
import {match} from 'ts-pattern';
import {GlmApiClient, GlmApiError} from './api';
import {AuthManager} from './auth';
import {GlmChatProvider} from './provider';

async function setApiKey(
  authManager: AuthManager,
  provider: GlmChatProvider,
): Promise<void> {
  await authManager.promptForApiKey();
  provider.fireLanguageModelChatInformationChange();
}

async function clearApiKey(
  authManager: AuthManager,
  provider: GlmChatProvider,
): Promise<void> {
  await authManager.deleteApiKey();
  provider.fireLanguageModelChatInformationChange();
  vscode.window.showInformationMessage('GLM API key cleared');
}

async function testConnection(
  authManager: AuthManager,
  provider: GlmChatProvider,
): Promise<void> {
  const key = await authManager.getApiKey();
  if (!key) {
    const shouldSetKey = await vscode.window.showInformationMessage(
      'No API key in extension storage. Use "GLM: Set API Key" first, then run this test again.',
      'Set API Key',
    );
    if (shouldSetKey === 'Set API Key') {
      await setApiKey(authManager, provider);
    }
    return;
  }

  const client = new GlmApiClient(key);
  try {
    await client.chat('glm-4.7', [{role: 'user', content: 'Ping'}], {
      maxTokens: 1,
    });
    vscode.window.showInformationMessage('GLM provider test succeeded.');
  } catch (error) {
    const message = match(error)
      .when(
        (value): value is GlmApiError =>
          value instanceof GlmApiError && value.statusCode === 401,
        () => 'Invalid API key. Please set a new key.',
      )
      .when(
        (value): value is Error => value instanceof Error,
        value => `GLM provider test failed: ${value.message}`,
      )
      .otherwise(value => `GLM provider test failed: ${String(value)}`);
    vscode.window.showErrorMessage(message);
  }
}

async function setThinkingEffort(): Promise<void> {
  const config = vscode.workspace.getConfiguration('glm-chat-provider');
  const current = config.get<string>('defaultThinkingMode', 'auto');

  const items = [
    {
      label: 'Auto',
      description: 'Let the model decide when to think',
      value: 'auto',
      picked: current === 'auto',
    },
    {
      label: 'Enabled',
      description: 'Always enable thinking mode',
      value: 'enabled',
      picked: current === 'enabled',
    },
    {
      label: 'Disabled',
      description: 'Always disable thinking mode',
      value: 'disabled',
      picked: current === 'disabled',
    },
  ];

  const choice = await vscode.window.showQuickPick(items, {
    placeHolder: 'Select thinking effort for GLM models',
  });

  if (!choice) {
    return;
  }

  await config.update('defaultThinkingMode', choice.value, true);
  vscode.window.showInformationMessage(
    `GLM thinking effort set to ${choice.label}`,
  );
}

export function activate(context: vscode.ExtensionContext): void {
  const authManager = new AuthManager(context.secrets);
  const provider = new GlmChatProvider(authManager);

  const manageActions: Record<string, () => Promise<void>> = {
    'Set API Key': () => setApiKey(authManager, provider),
    'Clear API Key': () => clearApiKey(authManager, provider),
    'Test Connection': () => testConnection(authManager, provider),
  };

  context.subscriptions.push(
    vscode.lm.registerLanguageModelChatProvider('zai', provider),
    vscode.commands.registerCommand('glm-chat-provider.setApiKey', async () => {
      await setApiKey(authManager, provider);
    }),
    vscode.commands.registerCommand(
      'glm-chat-provider.clearApiKey',
      async () => {
        await clearApiKey(authManager, provider);
      },
    ),
    vscode.commands.registerCommand('glm-chat-provider.manage', async () => {
      const choice = await vscode.window.showQuickPick(
        Object.keys(manageActions),
        {
          placeHolder: 'Manage Z.AI GLM provider',
        },
      );
      const action = choice ? manageActions[choice] : undefined;
      if (!action) {
        return;
      }
      await action();
    }),
    vscode.commands.registerCommand(
      'glm-chat-provider.setThinkingEffort',
      async () => {
        await setThinkingEffort();
      },
    ),
  );
}

export function deactivate(): void {}
