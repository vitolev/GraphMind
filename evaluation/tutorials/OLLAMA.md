# Running Ollama for GraphMind Evaluation

This guide explains how to set up and run Ollama for local LLM inference in the GraphMind pipeline.

## Quick Start

### 1. Install Ollama

```bash
brew install ollama
```

### 2. Start Ollama as a Background Service

**IMPORTANT:** Run Ollama as a background service so it keeps running even when you close the terminal.

```bash
brew services start ollama
```

This will:
- Start Ollama server on `http://localhost:11434`
- Automatically restart Ollama on system reboot
- Run in the background (no need to keep terminal open)

### 3. Verify Ollama is Running

Check if the service is running:
```bash
brew services list | grep ollama
```

You should see `started` next to `ollama`.

### 4. Download a Model

Pull the model you want to use (configured in `config/experiment_config.yaml`):

```bash
ollama pull llama3.2
```

This downloads the model to `~/.ollama/models/`. The first time will take a few minutes (the model is ~2GB).

**Recommended models:**
- `llama3.2` - Best balance of speed and quality (~1-2 sec/call) - **RECOMMENDED**
- `qwen2.5` - Great for math reasoning (~1-2 sec/call)
- `phi3` - Very fast (~0.5-1 sec/call)
- `tinyllama` - Fastest but lower quality (~0.3-0.5 sec/call)

### 5. Verify Model is Available

List installed models:
```bash
ollama list
```

Test the model:
```bash
ollama run llama3.2 "Solve: 2 + 2 = ?"
```

You should get a response quickly (within 1-2 seconds).

## Before Running the Pipeline

**ALWAYS verify Ollama is running before starting the pipeline:**

```bash
# Quick test
curl http://localhost:11434/api/tags
```

If you get a JSON response with models, Ollama is ready. If you get a connection error, start the service:

```bash
brew services start ollama
```

## Stopping Ollama

### Stop the Background Service

If you need to stop Ollama (e.g., to free up resources):

```bash
brew services stop ollama
```

### Verify It's Stopped

```bash
brew services list | grep ollama
```

Should show `stopped` status.

## Troubleshooting

### "ollama server not responding"

This means Ollama isn't running. Start it:
```bash
brew services start ollama
```

### "address already in use"

This means Ollama is already running (this is good!). You can verify with:
```bash
brew services list | grep ollama
```

### Model Not Found

If you get an error about the model not being found, pull it first:
```bash
ollama pull llama3.2
```

### Checking Ollama Logs

If you need to debug issues:
```bash
brew services info ollama
```

## Configuration

In `config/experiment_config.yaml`, ensure:

```yaml
llm_provider: ollama
ollama_model: llama3.2  # Match the model you pulled
ollama_base_url: http://localhost:11434  # Default, usually no need to change
```

## Performance Notes

- **First call** to a model may be slower (~5-10 seconds) as Ollama loads it into memory
- **Subsequent calls** are fast (~0.5-2 seconds depending on model and output length)
- Models are kept in memory for 5 minutes by default (configurable via `OLLAMA_KEEP_ALIVE` env var)
- Ollama automatically uses GPU acceleration if available (Apple Silicon MPS or NVIDIA CUDA)

## Running the Pipeline

Once Ollama is running, you can start the pipeline normally:

```bash
python main.py
```

The pipeline will automatically connect to Ollama and use the configured model.

