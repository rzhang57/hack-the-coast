"use strict";
const electron = require("electron");
electron.contextBridge.exposeInMainWorld("electronAPI", {
  minimize: () => electron.ipcRenderer.invoke("minimize"),
  close: () => electron.ipcRenderer.invoke("close"),
  toggleAlwaysOnTop: () => electron.ipcRenderer.invoke("toggle-always-on-top")
});
