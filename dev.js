if (typeof window === 'undefined') {
    // if node.js
    import('./dev-system.js')
        .then(async (module) => {
            await module.trainModel()
        })
        .catch((error) => {
            console.error('Failed to load the module:', error)
        })
} else {
    // if browsers
    const worker = new Worker(new URL('dev-browser.js', import.meta.url), {
        type: 'module'
    })
    worker.postMessage({ command: 'train' })
}
