import * as arrow from 'apache-arrow'
import wasmInit, { readParquet } from 'parquet-wasm'

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

export default class CosmopediaDataset {
    constructor(config) {
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
        const isBrowser =
            (typeof self !== 'undefined' &&
                typeof self.importScripts === 'function') ||
            typeof window !== 'undefined'
        if (isBrowser) await wasmInit()
        await this.fetchRandomShard()
    }

    async fetchRandomShard() {
        // this.disposeCurrentTable()
        const { slice, shards } = this.getWeightedRandomSlice(this.slices)
        const shardIndices = generatePaddedNumbers(0, shards, 5)
        const numShards = shardIndices.slice(-1)
        const allShards = shardIndices.slice(0, -1)
        const shard = randomValueFromArray(allShards)
        console.log('fetching shard:', `${shard}/${numShards}`, 'slice:', slice)
        const path = `data/${slice}/${this.split}-${shard}-of-${numShards}.parquet`
        this.url = `https://huggingface.co/datasets/${this.dataset}/resolve/main/${path}`
        console.log(this.url)
        try {
            const response = await fetch(this.url)
            this.buffer = new Uint8Array(await response.arrayBuffer())
            this.moveDataIntoTable()
            console.log('moved shard to table:', shard)
        } catch (err) {
            console.error(err)
            console.warn(
                `Failed to fetch shard (${shard}) from HuggingFace! We will continue using the old shard for now...`
            )
        }
    }

    moveDataIntoTable() {
        // Read Parquet buffer to Arrow Table
        // if (typeof this.arrowWasmTable?.free === 'function') {
        //     this.arrowWasmTable.free()
        //     this.arrowWasmTable = null
        // }
        this.arrowWasmTable = readParquet(this.buffer)
        // Convert to JS Arrow Table
        this.table = arrow.tableFromIPC(this.arrowWasmTable.intoIPCStream())
        // arrowWasmTable.free()
        this.buffer = null
    }

    // disposeCurrentTable() {
    //     if (this.table) {
    //         // Remove references to all batches
    //         this.table.batches.length = 0

    //         // Remove references to all columns
    //         if (this.table.schema && this.table.schema.fields) {
    //             this.table.schema.fields.forEach((field) => {
    //                 if (this.table[field.name]) {
    //                     this.table[field.name] = null
    //                 }
    //             })
    //         }

    //         // Remove reference to the schema
    //         this.table.schema = null

    //         // Remove the reference to the table itself
    //         this.table = null
    //     }

    //     // Suggest garbage collection if available
    //     if (global.gc) {
    //         global.gc()
    //     }
    // }

    getWeightedRandomSlice(slices) {
        // Calculate the total number of shards
        const totalShards = slices.reduce((sum, slice) => sum + slice.shards, 0)

        // Generate a random number between 0 and the total number of shards
        const randomShard = Math.floor(Math.random() * totalShards)

        // Find the slice that corresponds to the random shard
        let accumulatedShards = 0
        for (const slice of slices) {
            accumulatedShards += slice.shards
            if (randomShard < accumulatedShards) {
                return slice
            }
        }
    }

    viewSchema() {
        console.log(table.schema.toString())
    }

    loadSchema(array = [{ prompt: 'INPUT: ' }, { text: 'OUTPUT: ' }]) {
        this.schema = []
        array.map((obj) => {
            Object.entries(obj).forEach(([key, value]) => {
                const idx = this.table.schema.fields.findIndex(
                    (field) => field.name === key
                )
                console.assert(
                    idx !== -1,
                    `the key of ${key} does not exist in this dataset`
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
                let column
                try {
                    column = this.table.batches[batchIdx].getChildAt(obj.idx)
                } catch (err) {
                    console.error(err)
                    await this.fetchRandomShard()
                    return await this.fillCache()
                }
                if (rowIdx === null) {
                    rowIdx = randomBetween(0, column.length - 1)
                }
                const prefix = obj.value
                const data = column.get(rowIdx)
                // Some 'data' values appear to be random integers, with no other information. So, we
                // try to skip them here.
                if (/^-?\d+$/.test(data)) {
                    console.log(this.url)
                    console.log(data)
                    console.log('prefix was:', prefix)
                    console.log('batchIdx was:', batchIdx)
                    console.log('rowIdx was:', rowIdx)
                    console.log('data was:', data)
                    shouldSkip = true
                    // throw 'data was invalid' // this is temporary, for debugging, so we don't spam the terminal
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
