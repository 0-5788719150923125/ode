if (typeof window === 'undefined') {
    import('./dev-system.js')
        .then(async (module) => {
            await module.trainModel()
        })
        .catch((error) => {
            console.error('Failed to load the module:', error)
        })
} else {
    const worker = new Worker(new URL('dev-worker.js', import.meta.url), {
        type: 'module'
    })
    worker.postMessage({ command: 'train' })
}
