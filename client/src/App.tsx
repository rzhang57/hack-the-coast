import { useState, useRef, useEffect, useCallback } from 'react'
import {
  initAssist,
  sendMessage,
  getKeyStatus,
  saveApiKey,
  startBuffer,
  stopBuffer,
  getBufferStatus,
} from './services/api'
import './App.css'

interface Message {
  role: 'user' | 'assistant'
  text: string
}

type AppView = 'pill' | 'settings' | 'prompt' | 'chat'

const PILL = { w: 200, h: 48 }
const SETTINGS = { w: 300, h: 320 }
const PROMPT = { w: 320, h: 180 }
const CHAT = { w: 380, h: 520 }

const THRESHOLD_LIMIT = 5

function App() {
  const [view, setView] = useState<AppView>('pill')
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [pinned, setPinned] = useState(true)

  const [recording, setRecording] = useState(false)
  const [hasKey, setHasKey] = useState(false)
  const [editingKey, setEditingKey] = useState(false)
  const [apiKeyInput, setApiKeyInput] = useState('')
  const [threshold, setThreshold] = useState(0)

  const messagesEndRef = useRef<HTMLDivElement>(null)

  function resize(size: { w: number; h: number }) {
    window.electronAPI?.resizeWindow(size.w, size.h)
  }

  useEffect(() => {
    getKeyStatus().then(r => setHasKey(r.hasKey)).catch(() => {})
    getBufferStatus().then(r => setRecording(r.running)).catch(() => {})
  }, [])

  useEffect(() => {
    if (view !== 'pill' || !recording) return
    const interval = setInterval(async () => {
      try {
        const status = await getBufferStatus()
        setRecording(status.running)
        if (status.running) {
          // fetch threshold
        }
      } catch { /* server unavailable */ }
    }, 5000)
    return () => clearInterval(interval)
  }, [view, recording])

  useEffect(() => {
    if (threshold >= THRESHOLD_LIMIT && view === 'pill') {
      setView('prompt')
      resize(PROMPT)
      setThreshold(0)
    }
  }, [threshold, view])

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'instant' })
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

  function goToPill() {
    setView('pill')
    setMessages([])
    setInput('')
    resize(PILL)
  }

  function openSettings() {
    setView('settings')
    resize(SETTINGS)
    setEditingKey(false)
    getKeyStatus().then(r => setHasKey(r.hasKey)).catch(() => {})
  }

  async function handleSaveKey() {
    const key = apiKeyInput.trim()
    if (!key) return
    try {
      const res = await saveApiKey(key)
      setHasKey(res.hasKey)
      setEditingKey(false)
      setApiKeyInput('')
    } catch { /* handle error */ }
  }

  async function handleToggleBuffer() {
    try {
      if (recording) {
        await stopBuffer()
        setRecording(false)
        setThreshold(0)
      } else {
        await startBuffer()
        setRecording(true)
      }
    } catch { /* handle error */ }
  }

  function handleBlocked() {
    setView('chat')
    resize(CHAT)
    handleStuck()
  }

  function handleNotBlocked() {
    goToPill()
  }

  // -- PILL --
  if (view === 'pill') {
    return (
      <div className="drag flex items-center justify-center h-screen bg-transparent cursor-pointer" onClick={openSettings}>
        <div className="no-drag flex items-center gap-2 px-4 h-10 bg-[rgba(10,10,10,0.88)] border border-white/10 shadow-lg">
          <span className={`w-2 h-2 rounded-full shrink-0 ${recording ? 'bg-white dot-glow-white' : 'bg-white/30'}`} />
          <span className="text-xs font-semibold text-white/60 tracking-wide whitespace-nowrap">unstuck</span>
        </div>
      </div>
    )
  }

  // -- SETTINGS --
  if (view === 'settings') {
    return (
      <div className="flex flex-col h-screen bg-black overflow-hidden shadow-2xl border border-white/[0.08]">
        <div className="drag flex items-center justify-between h-9 px-3 bg-[rgba(15,15,15)] border-b border-white/[0.06] shrink-0">
          <span className="text-xs font-semibold text-white/70 tracking-wide">settings</span>
          <div className="no-drag flex gap-1">
            <button
              className="w-6 h-6 border-none bg-transparent text-white/50 cursor-pointer flex items-center justify-center text-xs p-0 hover:text-white"
              onClick={goToPill}
            >&#x2715;</button>
          </div>
        </div>

        <div className="flex-1 p-5 px-4 flex flex-col gap-6 overflow-y-auto">
          <div className="flex flex-col gap-2">
            <label className="text-[10px] font-semibold uppercase tracking-wider text-white/40">API Key</label>
            {hasKey && !editingKey ? (
              <div className="flex items-center gap-2">
                <span className="flex-1 text-xs text-white/70">●●●●●●●●</span>
                <button
                    className="h-8 px-3 border border-white/10 bg-transparent text-white/70 text-[11px] font-medium cursor-pointer shrink-0 hover:bg-white/10"
                  onClick={() => setEditingKey(true)}
                >Change</button>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <input
                  className="flex-1 h-8 px-2.5 border border-white/10 bg-white/[0.04] text-white/90 text-xs outline-none placeholder:text-white/30 focus:border-white/30"
                  type="password"
                  value={apiKeyInput}
                  onChange={e => setApiKeyInput(e.target.value)}
                  placeholder="Enter Gemini API key"
                />
                <button
                  className="h-8 px-3 border border-white/10 bg-transparent text-white/70 text-[11px] font-medium cursor-pointer shrink-0 hover:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed"
                  onClick={handleSaveKey}
                  disabled={!apiKeyInput.trim()}
                >Save</button>
              </div>
            )}
          </div>

          <div className="flex flex-col gap-2">
            <label className="text-[10px] font-semibold uppercase tracking-wider text-white/40">Focus Session</label>
            <button
              className={`flex items-center justify-center gap-2 w-full py-2.5 border text-white text-[13px] font-semibold cursor-pointer ${recording ? 'border-white/20 bg-white/10' : 'border-white/10 bg-white/[0.04]'} hover:bg-white/10`}
              onClick={handleToggleBuffer}
            >
              <span className={`w-2 h-2 rounded-full shrink-0 ${recording ? 'bg-white dot-glow-white' : 'bg-white/30'}`} />
              {recording ? 'Stop Recording' : 'Start Recording'}
            </button>
          </div>

          <div className="flex flex-col gap-2">
            <button
              className="flex items-center justify-center gap-2 w-full py-2.5 border border-white/10 bg-white text-black text-[13px] font-semibold cursor-pointer hover:bg-white/90"
              onClick={handleBlocked}
            >I'm stuck</button>
          </div>
        </div>
      </div>
    )
  }

  // -- PROMPT --
  if (view === 'prompt') {
    return (
      <div className="flex flex-col items-center justify-center h-screen p-6 bg-black shadow-2xl border border-white/[0.08] gap-5">
        <p className="text-sm font-semibold text-white/85 m-0 text-center">Are you feeling blocked?</p>
        <div className="flex gap-2.5 w-full">
          <button
            className="flex-1 py-2.5 border-none bg-white text-black text-xs font-semibold cursor-pointer hover:bg-white/90"
            onClick={handleBlocked}
          >Yes, I'm stuck</button>
          <button
            className="flex-1 py-2.5 border border-white/[0.12] bg-transparent text-white/70 text-xs font-semibold cursor-pointer hover:bg-white/10"
            onClick={handleNotBlocked}
          >No, I'm good</button>
        </div>
      </div>
    )
  }

  // -- CHAT --
  return (
    <div className="flex flex-col h-screen bg-black overflow-hidden shadow-2xl border border-white/[0.08]">
      <div className="drag flex items-center justify-between h-9 px-3 bg-[rgba(15,15,15)] border-b border-white/[0.06] shrink-0">
        <span className="text-xs font-semibold text-white/70 tracking-wide">unstuck</span>
        <div className="no-drag flex gap-1">
          <button
            className={`w-6 h-6 border-none bg-transparent cursor-pointer flex items-center justify-center text-xs p-0 hover:text-white ${pinned ? 'text-white' : 'text-white/50'}`}
            onClick={handleTogglePin}
            title={pinned ? 'Unpin' : 'Pin on top'}
          >&#x1F4CC;</button>
          <button
            className="w-6 h-6 border-none bg-transparent text-white/50 cursor-pointer flex items-center justify-center text-xs p-0 hover:text-white"
            onClick={() => window.electronAPI?.minimize()}
            title="Minimize"
          >&#x2013;</button>
          <button
            className="w-6 h-6 border-none bg-transparent text-white/50 cursor-pointer flex items-center justify-center text-xs p-0 hover:text-white"
            onClick={goToPill}
            title="Close"
          >&#x2715;</button>
        </div>
      </div>

      <div className="scrollbar-thin flex-1 overflow-y-auto p-4 px-3 flex flex-col gap-2.5">
        {messages.length === 0 && (
          <div className="flex items-center justify-center flex-1 text-center p-6">
            <p className="text-white/45 text-xs leading-relaxed m-0">Feeling stuck? Click the button below and I'll analyze your recent screen activity to help you get unblocked.</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] py-2 px-3 text-xs leading-relaxed whitespace-pre-wrap break-words ${
              msg.role === 'user'
                ? 'bg-white/10 text-white/90'
                : 'bg-white/[0.04] text-white/85'
            }`}>{msg.text}</div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="flex items-center gap-2 p-3 border-t border-white/[0.06] bg-[rgba(10,10,10,0.9)] shrink-0">
        {messages.length === 0 ? (
          <button
            className="w-full py-3 border-none bg-white text-black text-[13px] font-semibold cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed hover:bg-white/90"
            onClick={handleStuck}
            disabled={streaming}
          >{streaming ? 'Analyzing...' : "I'm stuck"}</button>
        ) : (
          <>
            <button
              className="w-9 h-9 border border-white/10 bg-transparent text-white/60 cursor-pointer text-base flex items-center justify-center shrink-0 p-0 hover:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed"
              onClick={handleStuck}
              disabled={streaming}
              title="Analyze screen again"
            >&#x1F504;</button>
            <input
              className="flex-1 h-9 px-3 border border-white/10 bg-white/[0.04] text-white/90 text-xs outline-none placeholder:text-white/30 focus:border-white/30 disabled:opacity-50"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a follow-up..."
              disabled={streaming}
            />
            <button
              className="w-9 h-9 border border-white/10 bg-white/10 text-white/80 cursor-pointer text-base flex items-center justify-center shrink-0 p-0 hover:bg-white/20 disabled:opacity-30 disabled:cursor-not-allowed"
              onClick={handleSend}
              disabled={streaming || !input.trim()}
            >&#x27A4;</button>
          </>
        )}
      </div>
    </div>
  )
}

export default App
