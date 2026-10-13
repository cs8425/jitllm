---
title: Privacy policy
description: jitllm runs models on your own computer and sends no personal data, telemetry or analytics anywhere.
---

**Effective date:** October 13, 2026

jitllm is an open-source program that runs AI language models entirely on your own computer. This policy explains what happens to your data when you use the jitllm app, its command-line tools and its local server.

## The short version

**jitllm does not collect, send or sell any personal data.** Your prompts, conversations, files and images are processed on your device and never leave it unless you send them somewhere yourself.

## What stays on your device

- **Prompts and responses:** all inference runs locally, on your CPU and GPU. Nobody else sees what you type or what the model answers.
- **Chat history:** saved chats are stored only in a folder on your computer. You can delete them at any time from the app or by deleting the folder.
- **Models:** model files you download or convert are stored on your computer.
- **Settings and caches:** app settings, compiled-kernel caches and prompt caches are stored locally.

## When jitllm connects to the internet

jitllm makes network requests only when you ask it to:

- **Downloading a model:** when you download or convert a model from Hugging Face, jitllm connects to huggingface.co to fetch the files. Hugging Face receives the request, including your IP address as with any download, under [its own privacy policy](https://huggingface.co/privacy). If you provide a Hugging Face access token, it is sent only to Hugging Face, to access the models you request.
- **Reporting a problem:** if jitllm crashes, it can open a pre-filled issue page on GitHub in your browser. Nothing is sent unless you choose to submit that issue yourself. GitHub's privacy statement applies to anything you post there.

jitllm sends **no telemetry, usage statistics, analytics or crash data automatically.**

## The local server

The jitllm background service (`jitllmd`) answers requests from programs on your computer and listens only on your computer by default. It does not forward your data to anyone. If you configure it to accept connections from other computers, data sent to it stays between those computers and yours.

## Children

jitllm does not knowingly collect any information from anyone, including children under 13.

## Changes

If this policy changes, the updated version is published here with a new effective date.

## Contact

Questions about this policy: open an issue at [github.com/jitllm/jitllm/issues](https://github.com/jitllm/jitllm/issues).
