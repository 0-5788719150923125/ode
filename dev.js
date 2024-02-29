;(async function () {
    if (typeof window === 'undefined') {
        // if node.js
        const module = await import('./dev-system.js')
        await module.trainModel()
    } else {
        // if browsers
        const worker = new Worker(new URL('dev-browser.js', import.meta.url), {
            type: 'module'
        })
        worker.postMessage({ command: 'train' })
    }
})()
