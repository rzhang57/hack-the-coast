interface ElectronAPI {
  minimize: () => Promise<void>
  close: () => Promise<void>
  toggleAlwaysOnTop: () => Promise<boolean>
  resizeWindow: (w: number, h: number) => Promise<void>
  animateResize: (w: number, h: number, durationMs: number) => Promise<void>
}

interface Window {
  electronAPI?: ElectronAPI
}
