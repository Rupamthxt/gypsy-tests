import * as vscode from "vscode";
import fetch from "node-fetch";

export function activate(context: vscode.ExtensionContext) {
  const disposable = vscode.commands.registerCommand("ai-testgen.generateTests", async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showErrorMessage("No active editor!");
      return;
    }

    const selection = editor.document.getText(editor.selection) || editor.document.getText();
    vscode.window.showInformationMessage("Generating unit tests...");

    try {
      const response = await fetch("http://localhost:8000/generate-tests", 
		{
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ code: selection })
		});
		const data = await response.json();
		const testCode = data.tests || "No output.";


      	editor.edit(editBuilder => {
        const position = editor.selection.end;
        editBuilder.insert(position, `\n\n${testCode}`);
        });
      //vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);

    } catch (err) {
      vscode.window.showErrorMessage(`Error generating tests: ${err}`);
    }
  });

  context.subscriptions.push(disposable);
}

export function deactivate() {}
