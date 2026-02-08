import { app, BrowserWindow, ipcMain, screen } from 'electron'
import { spawn, type ChildProcess } from 'child_process'
import path from 'path'
import fs from 'fs'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

let win: BrowserWindow | null = null
let flaskProcess: ChildProcess | null = null

const PILL_WIDTH = 200
const PILL_HEIGHT = 48

function findPythonVersion(venvLib: string): string {
  try {
    const dirs = fs.readdirSync(venvLib).filter(d => d.startsWith('python'))
    return dirs[0] || 'python3.12'
  } catch {
    return 'python3.12'
  }
}

function startFlask() {
  const isDev = !!process.env.VITE_DEV_SERVER_URL
  let serverDir: string
  let pythonPath: string
  let spawnEnv: Record<string, string | undefined>

  if (isDev) {
    serverDir = path.join(__dirname, '..', '..', 'server')
    pythonPath = path.join(serverDir, 'venv', 'bin', 'python')
    spawnEnv = { ...process.env, FLASK_ENV: 'development' }
  } else {
    serverDir = path.join(process.resourcesPath, 'server')
    const venvDir = path.join(serverDir, 'venv')
    const pyVersion = findPythonVersion(path.join(venvDir, 'lib'))
    const sitePackages = path.join(venvDir, 'lib', pyVersion, 'site-packages')

    pythonPath = path.join(venvDir, 'bin', 'python')

    spawnEnv = {
      ...process.env,
      FLASK_ENV: 'production',
      PYTHONPATH: sitePackages,
      VIRTUAL_ENV: venvDir,
      PATH: `/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin${process.env.PATH ? ':' + process.env.PATH : ''}`,
    }
  }

  console.log(`[flask] isDev=${isDev} serverDir=${serverDir}`)
  console.log(`[flask] pythonPath=${pythonPath}`)

  flaskProcess = spawn(pythonPath, ['-m', 'flask', '--app', 'app', 'run'], {
    cwd: serverDir,
    env: spawnEnv,
  })

  flaskProcess.stdout?.on('data', (data: Buffer) => {
    console.log(`[flask] ${data.toString().trim()}`)
  })

  flaskProcess.stderr?.on('data', (data: Buffer) => {
    console.log(`[flask] ${data.toString().trim()}`)
  })

  flaskProcess.on('error', (err) => {
    console.error('Failed to start Flask:', err)
  })
}

function createWindow() {
  const pos = getTargetBounds(PILL_WIDTH, PILL_HEIGHT)

  win = new BrowserWindow({
    width: PILL_WIDTH,
    height: PILL_HEIGHT,
    x: pos.x,
    y: pos.y,
    frame: false,
    transparent: true,
    resizable: false,
    alwaysOnTop: true,
    skipTaskbar: false,
    hasShadow: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.mjs'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  win.setContentProtection(false)

  if (process.platform === 'darwin') {
    win.setAlwaysOnTop(true, 'floating')
    win.setVisibleOnAllWorkspaces(true)
  }

  if (process.env.VITE_DEV_SERVER_URL) {
    win.loadURL(process.env.VITE_DEV_SERVER_URL)
  } else {
    win.loadFile(path.join(app.getAppPath(), 'dist', 'index.html'))
  }
}

app.whenReady().then(() => {
  startFlask()
  createWindow()
})

app.on('window-all-closed', () => {
  if (flaskProcess) {
    flaskProcess.kill()
    flaskProcess = null
  }
  app.quit()
})

app.on('before-quit', () => {
  if (flaskProcess) {
    flaskProcess.kill()
    flaskProcess = null
  }
})

ipcMain.handle('minimize', () => {
  win?.minimize()
})

ipcMain.handle('close', () => {
  win?.close()
})

ipcMain.handle('toggle-always-on-top', () => {
  if (!win) return false
  const next = !win.isAlwaysOnTop()
  if (process.platform === 'darwin') {
    win.setAlwaysOnTop(next, 'floating')
  } else {
    win.setAlwaysOnTop(next)
  }
  return next
})

function getTargetBounds(w: number, h: number) {
  const display = screen.getPrimaryDisplay()
  const area = display.workAreaSize
  return {
    x: Math.round((area.width - w) / 2),
    y: area.height - h - 24,
    width: w,
    height: h,
  }
}

ipcMain.handle('resize-window', (_event: unknown, w: number, h: number) => {
  if (!win) return
  const target = getTargetBounds(w, h)
  win.setResizable(true)
  win.setBounds(target, false)
  if (w === PILL_WIDTH && h === PILL_HEIGHT) win.setResizable(false)
})

let animTimer: ReturnType<typeof setInterval> | null = null

ipcMain.handle('animate-resize', (_event: unknown, w: number, h: number, durationMs: number) => {
  if (!win) return
  if (animTimer) { clearInterval(animTimer); animTimer = null }

  const start = win.getBounds()
  const end = getTargetBounds(w, h)
  const frames = Math.max(1, Math.round(durationMs / 16))
  let frame = 0

  win.setResizable(true)

  return new Promise<void>(resolve => {
    animTimer = setInterval(() => {
      frame++
      const t = Math.min(frame / frames, 1)
      const ease = 1 - Math.pow(1 - t, 3)

      win!.setBounds({
        x: Math.round(start.x + (end.x - start.x) * ease),
        y: Math.round(start.y + (end.y - start.y) * ease),
        width: Math.round(start.width + (end.width - start.width) * ease),
        height: Math.round(start.height + (end.height - start.height) * ease),
      }, false)

      if (frame >= frames) {
        clearInterval(animTimer!)
        animTimer = null
        win!.setBounds(end, false)
        if (w === PILL_WIDTH && h === PILL_HEIGHT) win!.setResizable(false)
        resolve()
      }
    }, 16)
  })
})
