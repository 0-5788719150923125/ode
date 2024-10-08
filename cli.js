import ODE from './src/index.js'

/**
 * These are default settings. Any one of these arguments can
 * be overwritten by passing them to the CLI, like this:
 * node cli.js --backend cpu --version 2 --batchSize 3
 */
const options = {
    action: 'train',
    backend: 'tensorflow',
    version: 4,
    batchSize: 1,
    gradientAccumulationSteps: 1,
    trainSteps: Infinity,
    seed: null,
    saveEvery: 0,
    generateEvery: 32,
    validateEvery: 0,
    validationSteps: 512,
    sampleLength: 256,
    'oversample1.5x': 0.0,
    oversample2x: 0.0,
    oversample4x: 0.0,
    predictLength: 128,
    doSample: true,
    temperature: 0.3,
    topK: 0,
    topP: 1.0,
    repetitionPenalty: 1.35,
    mirostat: false,
    mirostatState: {
        tau: 3.5, // target surprise
        eta: 0.15, // learning rate
        maxRepetition: 256, // max tokens to consider
        mu: 7.0 // initial mu (2 * tau)
    },
    corpus: 'https://www.gutenberg.org/files/100/old/shaks12.txt'
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
    const action = options.action
    delete options.action // else changes pollute the config hash

    const net = await ODE({
        contextLength: options.sampleLength,
        ...options
    })

    if (['infer', 'resume'].includes(action)) {
        await net.load()
    } else {
        await net.init()
    }

    if (['train', 'resume'].includes(action)) {
        let sampler
        const samplers = net.ode.samplers
        if (options.corpus.startsWith('http')) {
            sampler = samplers.HTTPSampler({ url: options.corpus })
        } else if (options.corpus === 'cosmopedia') {
            sampler = samplers.CosmopediaSampler(options)
        } else if (options.corpus === 'wikipedia') {
            sampler = samplers.WikipediaSampler(options)
        } else if (options.corpus === 'phi') {
            sampler = samplers.PhiSampler(options)
        } else if (options.corpus === 'refinedweb') {
            sampler = samplers.RefinedWebSampler(options)
        } else if (options.corpus === 'multi') {
            sampler = samplers.MultiSampler({
                samplers: [
                    samplers.CosmopediaSampler(),
                    samplers.DirectorySampler(
                        '/home/crow/Repos/vtx/lab/phi/train',
                        '\n\n'
                    )
                ]
            })
        } else if (options.corpus === 'balanced') {
            const rates = [1.0, 0.1, 0.5]
            sampler = samplers.WeightedSampler({
                samplers: [
                    samplers.CosmopediaSampler(options),
                    samplers.WikipediaSampler(options),
                    samplers.PhiSampler(options)
                ],
                rates: rates
            })
        } else if (options.corpus === 'devel') {
            const rates = [1.0, 0.25, 0.5]
            sampler = samplers.WeightedSampler({
                samplers: [
                    samplers.RefinedWebSampler(options),
                    samplers.CosmopediaSampler(options),
                    samplers.PhiSampler(options)
                ],
                rates: rates,
                seed: options.seed
            })
        } else {
            sampler = samplers.DirectorySampler({ directories: options.corpus })
        }

        const {
            ConsoleLogger,
            StateManager,
            InferenceGenerator,
            ValidationHandler,
            ModelSaver
        } = await import('./src/train.js')
        await net.train(sampler, options, [
            ConsoleLogger,
            StateManager,
            InferenceGenerator,
            ValidationHandler,
            ModelSaver
        ])
    } else if (action === 'infer') {
        const readline = (await import('readline')).default
        async function interactiveSession() {
            const rl = readline.createInterface({
                input: process.stdin,
                output: process.stdout
            })

            rl.question('PROMPT: ', async (text) => {
                const output = await net.generate({
                    ...options,
                    prompt: text,
                    maxNewTokens: 256
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
