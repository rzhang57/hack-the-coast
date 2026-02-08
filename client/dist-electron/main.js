import { app, ipcMain, BrowserWindow } from "electron";
import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";
const __dirname$1 = path.dirname(fileURLToPath(import.meta.url));
let win = null;
let flaskProcess = null;
function startFlask() {
  const isDev = !!process.env.VITE_DEV_SERVER_URL;
  let serverDir;
  let pythonPath;
  if (isDev) {
    serverDir = path.join(__dirname$1, "..", "..", "server");
    pythonPath = path.join(serverDir, "venv", "bin", "python");
  } else {
    serverDir = path.join(process.resourcesPath, "server");
    pythonPath = path.join(serverDir, "venv", "bin", "python");
  }
  console.log(`[flask] isDev=${isDev} serverDir=${serverDir}`);
  console.log(`[flask] pythonPath=${pythonPath}`);
  flaskProcess = spawn(pythonPath, ["-m", "flask", "--app", "app", "run"], {
    cwd: serverDir,
    env: { ...process.env, FLASK_ENV: "development" }
  });
  flaskProcess.stdout?.on("data", (data) => {
    console.log(`[flask] ${data.toString().trim()}`);
  });
  flaskProcess.stderr?.on("data", (data) => {
    console.log(`[flask] ${data.toString().trim()}`);
  });
  flaskProcess.on("error", (err) => {
    console.error("Failed to start Flask:", err);
  });
}
function createWindow() {
  win = new BrowserWindow({
    width: 380,
    height: 520,
    frame: false,
    transparent: false,
    resizable: true,
    alwaysOnTop: true,
    skipTaskbar: false,
    hasShadow: true,
    backgroundColor: "#1e1e1e",
    webPreferences: {
      preload: path.join(__dirname$1, "preload.mjs"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });
  if (process.platform === "darwin") {
    win.setAlwaysOnTop(true, "floating");
    win.setVisibleOnAllWorkspaces(true);
  }
  if (process.env.VITE_DEV_SERVER_URL) {
    win.loadURL(process.env.VITE_DEV_SERVER_URL);
  } else {
    win.loadFile(path.join(app.getAppPath(), "dist", "index.html"));
  }
}
app.whenReady().then(() => {
  startFlask();
  createWindow();
});
app.on("window-all-closed", () => {
  if (flaskProcess) {
    flaskProcess.kill();
    flaskProcess = null;
  }
  app.quit();
});
app.on("before-quit", () => {
  if (flaskProcess) {
    flaskProcess.kill();
    flaskProcess = null;
  }
});
ipcMain.handle("minimize", () => {
  win?.minimize();
});
ipcMain.handle("close", () => {
  win?.close();
});
ipcMain.handle("toggle-always-on-top", () => {
  if (!win) return false;
  const next = !win.isAlwaysOnTop();
  if (process.platform === "darwin") {
    win.setAlwaysOnTop(next, "floating");
  } else {
    win.setAlwaysOnTop(next);
  }
  return next;
});
