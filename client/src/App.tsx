import { useState, useRef, useEffect, useCallback } from 'react'
import { initAssist, sendMessage } from './services/api'
import './App.css'

interface Message {
  role: 'user' | 'assistant'
  text: string
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [pinned, setPinned] = useState(true)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  async function streamResponse(generator: AsyncGenerator<string>) {
    setStreaming(true)
    setMessages(prev => [...prev, { role: 'assistant', text: '' }])

    try {
      for await (const chunk of generator) {
        setMessages(prev => {
          const updated = [...prev]
          const last = updated[updated.length - 1]
          updated[updated.length - 1] = { ...last, text: last.text + chunk }
          return updated
        })
      }
    } finally {
      setStreaming(false)
    }
  }

  async function handleStuck() {
    if (streaming) return
    setMessages(prev => [...prev, { role: 'user', text: "I'm stuck" }])
    await streamResponse(initAssist())
  }

  async function handleSend() {
    const text = input.trim()
    if (!text || streaming) return
    setInput('')
    setMessages(prev => [...prev, { role: 'user', text }])
    await streamResponse(sendMessage(text))
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  async function handleTogglePin() {
    const result = await window.electronAPI?.toggleAlwaysOnTop()
    if (result !== undefined) setPinned(result)
  }

  return (
    <div className="chat-container">
      <div className="title-bar">
        <span className="title-text">unstuck</span>
        <div className="title-controls">
          <button
            className={`control-btn pin-btn ${pinned ? 'active' : ''}`}
            onClick={handleTogglePin}
            title={pinned ? 'Unpin' : 'Pin on top'}
          >
            &#x1F4CC;
          </button>
          <button
            className="control-btn"
            onClick={() => window.electronAPI?.minimize()}
            title="Minimize"
          >
            &#x2013;
          </button>
          <button
            className="control-btn close-btn"
            onClick={() => window.electronAPI?.close()}
            title="Close"
          >
            &#x2715;
          </button>
        </div>
      </div>

      <div className="messages">
        {messages.length === 0 && (
          <div className="empty-state">
            <p>Feeling stuck? Click the button below and I'll analyze your recent screen activity to help you get unblocked.</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="message-bubble">{msg.text}</div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-area">
        {messages.length === 0 ? (
          <button
            className="stuck-btn"
            onClick={handleStuck}
            disabled={streaming}
          >
            {streaming ? 'Analyzing...' : "I'm stuck"}
          </button>
        ) : (
          <>
            <button
              className="stuck-btn-small"
              onClick={handleStuck}
              disabled={streaming}
              title="Analyze screen again"
            >
              &#x1F504;
            </button>
            <input
              className="chat-input"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a follow-up..."
              disabled={streaming}
            />
            <button
              className="send-btn"
              onClick={handleSend}
              disabled={streaming || !input.trim()}
            >
              &#x27A4;
            </button>
          </>
        )}
      </div>
    </div>
  )
}

export default App
