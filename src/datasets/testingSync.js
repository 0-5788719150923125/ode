import fs from 'fs'
import { tableFromIPC } from 'apache-arrow'
import { readParquet } from 'parquet-wasm'

class CosmopediaDataset {
    constructor() {
        this.dataset = 'HuggingFaceTB/cosmopedia'
        this.slices = [
            { slice: 'auto_math_text', shards: 18 },
            { slice: 'khanacademy', shards: 1 },
            { slice: 'openstax', shards: 2 },
            { slice: 'stanford', shards: 13 },
            { slice: 'stories', shards: 43 },
            { slice: 'web_samples_v1', shards: 139 },
            { slice: 'web_samples_v2', shards: 118 },
            { slice: 'wikihow', shards: 2 }
        ]
        this.split = 'train'
        this.delimiter = '\n\n'
        this.eosToken = '<|eos|>'
        this.cacheSize = 20000
        this.cachedText = ''
        this.cycleShardInterval = 10000 // batches
        this.batches = 0
        this.arrowWasmTable = null
    }

    async init() {
        await this.fetchRandomShard()
    }

    async fetchRandomShard() {
        const { slice, shards } = randomValueFromArray(this.slices)
        const shardIndices = generatePaddedNumbers(0, shards, 5)
        const numShards = shardIndices.slice(-1)
        const allShards = shardIndices.slice(0, -1)
        const shard = randomValueFromArray(allShards)
        console.log('fetching shard:', `${shard}/${numShards}`, 'slice:', slice)
        const path = `data/${slice}/${this.split}-${shard}-of-${numShards}.parquet`
        this.url = `https://huggingface.co/datasets/${this.dataset}/resolve/main/${path}`
        console.log(this.url)
        const response = await fetch(this.url)
        this.buffer = new Uint8Array(await response.arrayBuffer())
        this.moveDataIntoTable()
        console.log('moved shard to table:', shard)
    }

    moveDataIntoTable() {
        // Read Parquet buffer to Arrow Table
        this.arrowWasmTable = readParquet(this.buffer)
        // Get the Arrow stream buffer
        this.arrowStreamBuffer = this.arrowWasmTable.intoIPCStream()
        // Convert to JS Arrow Table
        this.table = tableFromIPC(this.arrowStreamBuffer)
    }

    loadSchema(array = [{ prompt: 'INPUT: ' }, { text: 'OUTPUT: ' }]) {
        this.schema = []
        array.map((obj) => {
            Object.entries(obj).forEach(([key, value]) => {
                const idx = this.table.schema.fields.findIndex(
                    (field) => field.name === key
                )
                this.schema.push({ idx, value })
            })
        })
    }

    async fillCache() {
        while (this.cachedText.length < this.cacheSize) {
            let shouldSkip = false
            let batchIdx = randomBetween(0, this.table.batches.length - 1)

            const text = []

            let rowIdx = null
            for (const obj of this.schema) {
                let column = this.table.batches[batchIdx].getChildAt(obj.idx)
                if (rowIdx === null) {
                    rowIdx = randomBetween(0, column.length - 1)
                }
                const prefix = obj.value
                const data = column.get(rowIdx)
                // Some 'data' values appear to be random integers, with no other information.
                // We handle that here.
                if (/^-?\d+$/.test(data)) {
                    console.log(
                        'FAILED TO PARSE SHARD: Received a BigInt instead of text.'
                    )
                    console.log(data)
                    console.log('prefix was:', prefix)
                    console.log('batchIdx was:', batchIdx)
                    console.log('rowIdx was:', rowIdx)
                    console.log('data was:', data)
                    shouldSkip = true
                    fs.writeFileSync(
                        './arrowStreamBuffer.txt',
                        this.arrowStreamBuffer
                    )
                    await this.fetchRandomShard()
                }
                text.push(prefix + data)
            }
            if (shouldSkip) continue
            this.cachedText += text.join(this.delimiter) + this.eosToken
        }
    }

    async getSample(size = 512) {
        this.batches++
        if (this.batches % this.cycleShardInterval === 0) {
            await this.fetchRandomShard()
        }
        await this.fillCache()
        const sample = this.cachedText.slice(0, size)
        this.cachedText = this.cachedText.slice(size)
        return sample
    }
}

function randomBetween(min, max) {
    return Math.floor(Math.random() * (max - min + 1) + min)
}

function randomValueFromArray(array) {
    const randomIndex = Math.floor(Math.random() * array.length)
    return array[randomIndex]
}

function generatePaddedNumbers(start, end, totalDigits) {
    const numbers = []
    for (let i = start; i <= end; i++) {
        numbers.push(String(i).padStart(totalDigits, '0'))
    }
    return numbers
}

const numIterations = 1_000_000_000_000
async function main() {
    const sampler = new CosmopediaDataset()
    await sampler.init()
    sampler.loadSchema([{ prompt: 'INPUT: ' }, { text: 'OUTPUT: ' }])
    for (let i = 0; i < numIterations; i++) {
        if (i % 1000 === 0) console.log('iterations:', i)
        await sampler.getSample()
    }
}

main()
