import { useState, useRef, useEffect, useCallback } from 'react'
import {
  initAssist,
  sendMessage,
  getApiKey,
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
const SETTINGS = { w: 300, h: 250 }
const CHAT = { w: 380, h: 520 }

const THRESHOLD_LIMIT = 5

function App() {
  const [view, setView] = useState<AppView>('pill')
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)

  const [recording, setRecording] = useState(false)
  const [apiKeyInput, setApiKeyInput] = useState('')
  const [showKey, setShowKey] = useState(false)
  const [threshold, setThreshold] = useState(0)
  const [promptFading, setPromptFading] = useState(false)

  const saveTimerRef = useRef<ReturnType<typeof setTimeout>>(null)

  const messagesEndRef = useRef<HTMLDivElement>(null)

  function resize(size: { w: number; h: number }) {
    window.electronAPI?.resizeWindow(size.w, size.h)
  }

  function animateResize(size: { w: number; h: number }, ms = 1000) {
    return window.electronAPI?.animateResize(size.w, size.h, ms)
  }

  useEffect(() => {
    getApiKey().then(r => setApiKeyInput(r.key)).catch(() => {})
    getBufferStatus().then(r => setRecording(r.running)).catch(() => {})
  }, [])

  useEffect(() => {
    if (view !== 'pill' || !recording) return
    const interval = setInterval(async () => {
      try {
        const status = await getBufferStatus()
        setRecording(status.running)
        if (status.running) {
          setThreshold((v) => v + 1)
        }
      } catch { /* server unavailable */ }
    }, 1000)
    return () => clearInterval(interval)
  }, [view, recording])

  useEffect(() => {
    // TODO: threshold will become a boolean state instead, play ping sound
    if (threshold >= THRESHOLD_LIMIT && view === 'pill') {
      setThreshold(0)
      // setPromptFading(false)
      // setView('prompt')
      // animateResize(PROMPT, 400)
      handleBlocked()
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

  function goToPill() {
    setView('pill')
    setMessages([])
    setInput('')
    resize(PILL)
  }

  function openSettings() {
    setView('settings')
    resize(SETTINGS)
    getApiKey().then(r => setApiKeyInput(r.key)).catch(() => {})
  }

  function handleApiKeyChange(value: string) {
    setApiKeyInput(value)
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
    saveTimerRef.current = setTimeout(() => {
      saveApiKey(value.trim()).catch(() => {})
    }, 500)
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
        goToPill()
      }
    } catch { /* handle error */ }
  }

  function handleBlocked() {
    setView('chat')
    animateResize(CHAT)
    handleStuck()
  }

  async function handleNotBlocked() {
    setPromptFading(true)
    await new Promise(r => setTimeout(r, 300))
    setView('pill')
    setMessages([])
    setInput('')
    await animateResize(PILL, 500)
  }

  // -- PILL --
  if (view === 'pill') {
    return (
      <div className="drag flex items-center justify-center h-screen bg-transparent cursor-pointer" onClick={openSettings}>
        <div className="no-drag flex items-center gap-2 px-4 h-10 bg-[rgba(10,10,10,0.88)] rounded-full border border-white/10">
          <span
              className={`w-2 h-2 rounded-full shrink-0 ${recording ? 'bg-white animate-pulse shadow-[0_0_8px_rgba(239,68,68,0.6)]' : 'bg-white/30'}`}/>
          <span className="text-xs font-semibold text-white/60 tracking-wide whitespace-nowrap">focus</span>
        </div>
      </div>
    )
  }

  if (view === 'settings') {
    return (
      <div className="flex flex-col h-screen bg-[rgba(10,10,10,0.88)] overflow-hidden shadow-2xl border border-white/[0.08]">
        <div className="drag flex items-center justify-between h-9 px-3 shrink-0">
          <span className="text-xs font-semibold text-white/70 tracking-tight">Settings</span>
          <div className="no-drag flex gap-1">
            <button
              className="w-6 h-6 border-none bg-transparent text-white/50 cursor-pointer flex items-center justify-center p-0 hover:text-white"
              onClick={goToPill}
            ><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round"><line x1="5" y1="12" x2="19" y2="12"/></svg></button>
          </div>
        </div>

        <div className="flex-1 p-5 px-4 flex flex-col gap-2 overflow-y-auto">
          <div className="flex flex-col gap-2">
            <label className="text-[10px] font-semibold tracking-tight text-white/40">Gemini API Key</label>
            <div className="relative">
              <input
                className="w-full h-8 px-2.5 pr-8 border border-white/10 bg-white/[0.04] text-white/90 text-xs outline-none placeholder:text-white/30 focus:border-white/30"
                type={showKey ? 'text' : 'password'}
                value={apiKeyInput}
                onChange={e => handleApiKeyChange(e.target.value)}
                placeholder="Enter Gemini API key"
                spellCheck={false}
              />
              <button
                className="absolute right-0 top-0 w-8 h-8 border-none bg-transparent text-white/40 cursor-pointer flex items-center justify-center p-0 hover:text-white/80"
                onClick={() => setShowKey(v => !v)}
                title={showKey ? 'Hide' : 'Show'}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  {showKey ? (
                    <>
                      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
                      <circle cx="12" cy="12" r="3"/>
                    </>
                  ) : (
                    <>
                      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"/>
                      <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"/>
                      <path d="M14.12 14.12a3 3 0 1 1-4.24-4.24"/>
                      <line x1="1" y1="1" x2="23" y2="23"/>
                    </>
                  )}
                </svg>
              </button>
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <label className="text-[10px] font-semibold tracking-tight text-white/40">Focus Session</label>
            <button
              className={`flex items-center justify-center gap-2 w-full py-2.5 border text-white text-[13px] font-semibold cursor-pointer ${recording ? 'border-white/20 bg-white/10' : 'border-white/10 bg-white/[0.04]'} hover:bg-white/10`}
              onClick={handleToggleBuffer}
            >
              <span className={`w-2 h-2 rounded-full shrink-0 ${recording ? 'bg-white dot-glow-white' : 'bg-white/30'}`} />
              {recording ? 'End Session' : 'Start Session'}
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
      <div className="flex flex-col items-center justify-center h-screen p-6 bg-[rgba(10,10,10,0.88)] shadow-2xl border border-white/[0.08] gap-5">
        <div className={`flex flex-col items-center gap-5 w-full ${promptFading ? 'prompt-fade-out' : 'prompt-fade-in'}`}>
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
      </div>
    )
  }

  // -- CHAT --
  return (
    <div className="flex flex-col h-screen bg-[rgba(10,10,10,0.88)] overflow-hidden shadow-2xl border-white/[0.08]">
      <div className="drag flex items-center justify-between h-9 px-3 border-white/[0.06] shrink-0">
        <span className="text-xs font-semibold text-white/70 tracking-tight">spot agent</span>
        <div className="no-drag flex gap-1">
          <button
            className="w-6 h-6 border-none bg-transparent text-white/50 cursor-pointer flex items-center justify-center p-0 hover:text-white"
            onClick={goToPill}
            title="Minimize"
          ><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round"><line x1="5" y1="12" x2="19" y2="12"/></svg></button>
        </div>
      </div>

      <div className="scrollbar-thin flex-1 overflow-y-auto p-4 px-3 flex flex-col gap-2.5">
        {messages.length === 0 && (
          <div className="flex items-center justify-center flex-1 text-center p-6">
            <p className="text-white/45 text-xs leading-relaxed m-0">Feeling stuck? Click the button below and I'll analyze your recent screen activity to help you get unblocked.</p>
          </div>
        )}
        {messages.map((msg, i) => {
          const isThinking = streaming && msg.role === 'assistant' && msg.text === '' && i === messages.length - 1
          return (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[85%] py-2 px-3 text-xs leading-relaxed whitespace-pre-wrap break-words ${
                msg.role === 'user'
                  ? 'bg-white/10 text-white/90'
                  : 'bg-white/[0.04] text-white/85'
              }`}>
                {isThinking ? (
                  <div className="flex items-center gap-1.5 py-0.5">
                    <div className="thinking-dot" />
                    <div className="thinking-dot" />
                    <div className="thinking-dot" />
                  </div>
                ) : msg.text}
              </div>
            </div>
          )
        })}
        <div ref={messagesEndRef} />
      </div>

      <div className="flex items-center gap-2 p-3 border-t border-white/[0.06] shrink-0">
        {messages.length === 0 ? (
          <button
            className="w-full py-3 border-none bg-white text-black text-[13px] font-semibold cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed hover:bg-white/90"
            onClick={handleStuck}
            disabled={streaming}
          >{streaming ? 'Analyzing...' : "I'm stuck"}</button>
        ) : (
          <>
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
