interface ElectronAPI {
  minimize: () => Promise<void>
  close: () => Promise<void>
  toggleAlwaysOnTop: () => Promise<boolean>
}

interface Window {
  electronAPI?: ElectronAPI
}
