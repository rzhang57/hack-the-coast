"use strict";
const electron = require("electron");
electron.contextBridge.exposeInMainWorld("electronAPI", {
  minimize: () => electron.ipcRenderer.invoke("minimize"),
  close: () => electron.ipcRenderer.invoke("close"),
  toggleAlwaysOnTop: () => electron.ipcRenderer.invoke("toggle-always-on-top"),
  resizeWindow: (w, h) => electron.ipcRenderer.invoke("resize-window", w, h),
  animateResize: (w, h, durationMs) => electron.ipcRenderer.invoke("animate-resize", w, h, durationMs)
});
