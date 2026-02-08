import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('electronAPI', {
  minimize: () => ipcRenderer.invoke('minimize'),
  close: () => ipcRenderer.invoke('close'),
  toggleAlwaysOnTop: () => ipcRenderer.invoke('toggle-always-on-top'),
})
