const API_BASE = 'http://localhost:5000'

async function* readSSE(response: Response): AsyncGenerator<string> {
  const reader = response.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed.startsWith('data: ')) continue

      const payload = trimmed.slice(6)
      if (payload === '[DONE]') return

      try {
        const parsed = JSON.parse(payload)
        if (parsed.text) yield parsed.text
      } catch {
        // skip malformed lines
      }
    }
  }
}

export async function* initAssist(): AsyncGenerator<string> {
  const response = await fetch(`${API_BASE}/assist`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  })
  yield* readSSE(response)
}

export async function* sendMessage(message: string): AsyncGenerator<string> {
  const response = await fetch(`${API_BASE}/assist/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  })
  yield* readSSE(response)
}
