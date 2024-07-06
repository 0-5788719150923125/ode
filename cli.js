import ODE from './src/index.js'

/**
 * Any one of these arguments can be passed to the CLI like this:
 * node cli.js --backend cpu --version 2 --batchSize 3
 */
const options = {
    mode: 'train',
    backend: 'tensorflow',
    version: 3,
    batchSize: 1,
    gradientAccumulationSteps: 1,
    generateEvery: 256,
    sampleLength: 256,
    predictLength: 128,
    saveEvery: 0,
    corpus: null
}

// Get the command-line arguments
const args = process.argv.slice(2)

// Do nothing is no arguments are provided
if (args.length === 0) {
    console.error('ERROR: You must pass at least 1 argument to this script!')
    console.log('node cli.js \\')
    for (const [key, value] of Object.entries(options)) {
        console.log(`  --${key} ${value} \\`)
    }
    process.exit()
}

// Parse named arguments
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
    console.log('loading corpus:', options.corpus)
    if (options.corpus.startsWith('http')) {
        corpus = await net.ode.samplers.fetchURLSampler(
            options.corpus,
            options.corpus.split('/')[-1]
        )
    } else if (options.corpus === 'cosmopedia') {
        // pass
    } else {
        corpus = await net.ode.samplers.directorySampler(options.corpus, '\n\n')
    }

    if (['infer', 'continue'].includes(options.mode)) {
        await net.load()
    } else {
        await net.init()
    }

    if (['train', 'continue'].includes(options.mode)) {
        let dataset
        if (options.samplingMethod === 'sequential') {
            dataset = net.ode.samplers.sequentialStringSampler(
                options.sampleLength,
                corpus
            )
        } else if (options.corpus === 'cosmopedia') {
            dataset = net.ode.samplers.CosmopediaSampler(
                options.sampleLength * 10
            )
        } else {
            dataset = net.ode.samplers.stringSampler(
                options.sampleLength * 8,
                0,
                corpus
            )
        }

        const {
            ConsoleLogger,
            // MetricsCollector,
            PredictionSampler,
            ModelSaver
        } = await import('./src/train.js')
        await net.train(dataset, options, [
            ConsoleLogger,
            // MetricsCollector,
            PredictionSampler,
            ModelSaver
        ])
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
                    temperature: 0.45,
                    maxNewTokens: 256,
                    repetitionPenalty: 1.1
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
    console.clear()
    await orchestrate(options)
})()
