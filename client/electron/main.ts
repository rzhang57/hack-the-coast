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

function getPillPosition() {
  const display = screen.getPrimaryDisplay()
  const { width, height } = display.workAreaSize
  return {
    x: Math.round((width - PILL_WIDTH) / 2),
    y: height - PILL_HEIGHT - 24,
  }
}

function createWindow() {
  const pos = getPillPosition()

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

  win.setContentProtection(true)

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

ipcMain.handle('resize-window', (_event: unknown, w: number, h: number) => {
  if (!win) return
  if (w === PILL_WIDTH && h === PILL_HEIGHT) {
    const pos = getPillPosition()
    win.setBounds({ x: pos.x, y: pos.y, width: w, height: h }, false)
    win.setResizable(false)
  } else {
    const display = screen.getPrimaryDisplay()
    const area = display.workAreaSize
    const x = Math.round((area.width - w) / 2)
    const y = Math.round((area.height - h) / 2)
    win.setResizable(true)
    win.setBounds({ x, y, width: w, height: h }, false)
  }
})
