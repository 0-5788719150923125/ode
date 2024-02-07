const worker = new Worker(new URL('dev-worker.js', import.meta.url), {
    type: 'module'
})
worker.postMessage({ command: 'train' })
