import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('electronAPI', {
  minimize: () => ipcRenderer.invoke('minimize'),
  close: () => ipcRenderer.invoke('close'),
  toggleAlwaysOnTop: () => ipcRenderer.invoke('toggle-always-on-top'),
  resizeWindow: (w: number, h: number) => ipcRenderer.invoke('resize-window', w, h),
  animateResize: (w: number, h: number, durationMs: number) => ipcRenderer.invoke('animate-resize', w, h, durationMs),
})
