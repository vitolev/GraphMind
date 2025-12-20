# Multiple Groq API Keys Setup

This guide explains how to use multiple Groq API keys for automatic rotation when rate limits are hit.

## How It Works

The system automatically:
1. **Rotates between API keys** when rate limits are encountered
2. **Switches models** when returning to a previously rate-limited API key (to avoid hitting the same limit again)
3. **Tracks rate-limited API keys** and avoids them until they're available again

## Setup Instructions

### Step 1: Create/Edit `.env` File

Create a `.env` file in the project root (if it doesn't exist) or edit the existing one.

### Step 2: Add Your API Keys

Add your API keys as a **comma-separated list**:

```env
GROQ_API_KEYS=gsk_your_first_key,gsk_your_second_key,gsk_your_third_key
```

**Example:**
```env
GROQ_API_KEYS=gsk_abc123...,gsk_def456...,gsk_ghi789...
```

### Step 3: Verify Setup

When you run the pipeline, you should see:
```
✓ Initialized Groq API key 1/3
✓ Initialized Groq API key 2/3
✓ Initialized Groq API key 3/3
✓ Initialized 3 Groq API key(s)
```

## How Rotation Works

### When Rate Limit is Hit:

1. **API Key Rotation**: The system marks the current API key as rate-limited and automatically switches to the next available API key.

2. **Model Rotation**: When switching back to a previously rate-limited API key, the system also switches to a different model to avoid hitting the same rate limit again.

3. **Automatic Recovery**: When all API keys are rate-limited, the system resets and tries all keys again.

### Example Flow:

```
Request 1: API Key 1 + Model A → Success
Request 2: API Key 2 + Model B → Success
Request 3: API Key 1 + Model C → Rate Limited!
           → Switch to API Key 2 + Model D
Request 4: API Key 2 + Model D → Success
Request 5: API Key 3 + Model A → Success
...
(When API Key 1 becomes available again)
Request N: API Key 1 + Model E → Success (different model to avoid same limit)
```

## Benefits

- **Higher Throughput**: With N API keys, you get N × 14.4K requests per day
- **Automatic Failover**: No manual intervention needed when rate limits are hit
- **Smart Model Selection**: Avoids repeating the same model on a previously rate-limited API key
- **Secure**: API keys stored in `.env` file (already in `.gitignore`)

## Security Notes

- **Never commit `.env` file** to version control (it's already in `.gitignore`)
- Keep your API keys secure and don't share them
- Each API key should have its own Groq account/plan

## Troubleshooting

### "No Groq API keys found"
- Make sure your `.env` file is in the project root
- Check that `GROQ_API_KEYS` is set correctly (comma-separated, no spaces around commas)
- Verify the keys don't have extra spaces or quotes

### "All API keys are rate-limited"
- Wait for the rate limit to reset (usually per day)
- Consider adding more API keys
- Check your Groq account limits

### Keys not loading
- Make sure `python-dotenv` is installed: `pip install python-dotenv`
- Restart your Python process after changing `.env` file

