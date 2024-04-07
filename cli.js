import ODE from './src/index.js'

/**
 * Any one of these arguments can be passed to this CLI like this:
 * node cli.js --backend cpu --version 2 --batchSize 3
 */
const options = {
    mode: 'train',
    backend: 'tensorflow',
    version: 3,
    batchSize: 1,
    gradientAccumulationSteps: 128,
    generateEvery: 256,
    sampleLength: 256,
    predictLength: 128,
    corpus: null,
    debug: false
}

// Get the command-line arguments
const args = process.argv.slice(2)

// Parse the named arguments
for (let i = 0; i < args.length; i++) {
    if (args[i].startsWith('--')) {
        const key = args[i].slice(2)
        const value = args[i + 1]

        // Check if the value is a number
        if (/^\d+$/.test(value)) {
            options[key] = parseInt(value, 10)
        } else if (/^\d+\.\d+$/.test(value)) {
            options[key] = parseFloat(value)
        } else {
            options[key] = value
        }

        i++
    }
}

// Load the model and do things with it
async function orchestrate(options) {
    const net = await ODE({
        contextLength: options.sampleLength,
        clipValue: 1.0,
        ...options
    })

    let corpus
    if (options.corpus) {
        const gun = net.ode.samplers.gunSampler()
        await gun.init()
        await gun.uploadDirectory('custom', options.corpus)
        corpus = await gun.getDataset('custom')
    }

    if (['infer', 'continue'].includes(options.mode)) {
        await net.load()
    } else {
        await net.init()
    }

    if (['train', 'continue'].includes(options.mode)) {
        const dataset = net.ode.samplers.stringSampler(
            options.sampleLength * 5,
            0,
            corpus
        )
        await net.train(dataset, options)
    } else if (options.mode === 'infer') {
        const readline = (await import('readline')).default
        async function interactiveSession() {
            const rl = readline.createInterface({
                input: process.stdin,
                output: process.stdout
            })

            rl.question('PROMPT: ', async (text) => {
                const output = await net.generate({
                    prompt: text,
                    doSample: true,
                    temperature: 0.7,
                    topK: 4,
                    maxNewTokens: 128
                })
                console.log(`OUTPUT: ${output}`)
                rl.close()
                await interactiveSession()
            })
        }
        await interactiveSession()
    }
}

;(async function () {
    await orchestrate(options)
})()
