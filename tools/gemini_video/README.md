# Gemini Video

## Overview

The **Gemini Video Tool** accesses Gemini's current video generation models. It keeps the existing Dify tool interface while routing each model to the API it requires.

- Gemini Omni 1.1 Flash (recommended)
- Veo 3.1
- Veo 3.1 Fast
- Veo 3.1 Lite

Use the exact model IDs shown in the tool selector. `gemini-omni-1.1-flash` uses the Interactions API; the Veo models use the long-running `generate_videos` API.

Gemini Omni does not expose dedicated duration or negative-prompt fields. The tool preserves those Dify inputs by adding them to the regular prompt. Omni supports 360p, 720p, 1080p, and 4K output. Veo validates its stricter combinations (for example, 1080p/4K and reference-image generation require an 8-second duration).

---

## Configuration

### 1. Get a Gemini API Key

Go to the [Google AI Studio](https://aistudio.google.com/app/apikey) and make a Gemini API Key.

### 2. Gemini Video Tool in Dify

1. In the **Dify Console**, go to **Plugin Marketplace**.  
2. Search for **Gemini Video** and install it.

### 3. Configure in Dify

In **Dify Console > Tools > Gemini Video > Authorize**, enter:  

- **Gemini API Key**: The key from [Google AI Studio](https://aistudio.google.com/app/apikey).
---

## Usage

The Gemini Video Tool can be used in the following application types:

### Chatflow / Workflow Applications
Add a **Gemini Video node** to generate or edit videos during flow execution.

### Agent Applications
Enable the **Gemini Video tool** in Agent applications.  
