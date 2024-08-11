import ODE from './src/index.js'

/**
 * Any one of these arguments can be passed to the CLI like this:
 * node cli.js --backend cpu --version 2 --batchSize 3
 */
const options = {
    action: 'train',
    backend: 'tensorflow',
    version: 3,
    batchSize: 1,
    gradientAccumulationSteps: 1,
    generateEvery: 256,
    sampleLength: 256,
    predictLength: 128,
    saveEvery: 0,
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
    const net = await ODE({
        contextLength: options.sampleLength,
        ...options
    })

    if (['infer', 'resume'].includes(options.action)) {
        await net.load()
    } else {
        await net.init()
    }

    if (['train', 'resume'].includes(options.action)) {
        let sampler
        const samplers = net.ode.samplers
        if (options.corpus.startsWith('http')) {
            sampler = samplers.HTTPSampler(options.corpus)
        } else if (options.corpus === 'cosmopedia') {
            sampler = samplers.CosmopediaSampler()
        } else if (options.corpus === 'wikipedia') {
            sampler = samplers.WikipediaSampler()
        } else if (options.corpus === 'multi') {
            sampler = samplers.MultiSampler([
                samplers.CosmopediaSampler(),
                samplers.DirectorySampler(
                    '/home/crow/Repos/vtx/lab/phi/train',
                    '\n\n'
                )
            ])
        } else if (options.corpus === 'balanced') {
            const rates = [1.0, 0.5]
            sampler = samplers.WeightedSampler(
                [samplers.CosmopediaSampler(), samplers.WikipediaSampler()],
                rates
            )
        } else {
            sampler = samplers.DirectorySampler(options.corpus, '\n\n')
        }

        const {
            ConsoleLogger,
            // MetricsCollector,
            PredictionSampler,
            ModelSaver
        } = await import('./src/train.js')
        await net.train(sampler, options, [
            ConsoleLogger,
            // MetricsCollector,
            PredictionSampler,
            ModelSaver
        ])
    } else if (options.action === 'infer') {
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
