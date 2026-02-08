import { app, BrowserWindow, ipcMain } from 'electron'
import { spawn, type ChildProcess } from 'child_process'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

let win: BrowserWindow | null = null
let flaskProcess: ChildProcess | null = null

function startFlask() {
  const serverDir = path.join(__dirname, '..', '..', 'server')
  const pythonPath = path.join(serverDir, 'venv', 'bin', 'python')

  flaskProcess = spawn(pythonPath, ['-m', 'flask', '--app', 'app', 'run'], {
    cwd: serverDir,
    env: { ...process.env, FLASK_ENV: 'development' },
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
  win = new BrowserWindow({
    width: 380,
    height: 520,
    frame: false,
    transparent: true,
    resizable: true,
    alwaysOnTop: true,
    skipTaskbar: false,
    hasShadow: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.mjs'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  if (process.platform === 'darwin') {
    win.setAlwaysOnTop(true, 'floating')
    win.setVisibleOnAllWorkspaces(true)
  }

  if (process.env.VITE_DEV_SERVER_URL) {
    win.loadURL(process.env.VITE_DEV_SERVER_URL)
  } else {
    win.loadFile(path.join(__dirname, '..', 'dist', 'index.html'))
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
