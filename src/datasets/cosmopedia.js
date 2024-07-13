import * as arrow from 'apache-arrow'
import wasmInit, { readParquet } from 'parquet-wasm'
import {
    generatePaddedNumbers,
    randomBetween,
    randomValueFromArray
} from '../utils.js'

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
        this.cycleShardInterval = 10000
        this.batches = 0
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
        const { slice, shards } = this.getWeightedRandomSlice(this.slices)
        const shardIndices = generatePaddedNumbers(0, shards, 5)
        const numShards = shardIndices.slice(-1)
        const allShards = shardIndices.slice(0, -1)

        const shard = randomValueFromArray(allShards)
        console.log('fetching shard:', `${shard}/${numShards}`, 'slice:', slice)
        const path = `data/${slice}/${this.split}-${shard}-of-${numShards}.parquet`
        const url = `https://huggingface.co/datasets/${this.dataset}/resolve/main/${path}`
        console.log(url)
        try {
            const response = await fetch(url)
            this.buffer = new Uint8Array(await response.arrayBuffer())
            this.moveDataIntoTable()
            console.log('moved shard to table:', shard)
        } catch (err) {
            console.warn(
                `Failed to fetch shard (${shard}) from HuggingFace! We will continue using the old shard for now...`
            )
        }
    }

    moveDataIntoTable() {
        // Read Parquet buffer to Arrow Table
        const arrowWasmTable = readParquet(this.buffer)
        // Convert to JS Arrow Table
        this.table = arrow.tableFromIPC(arrowWasmTable.intoIPCStream())
    }

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

    loadSchema(array) {
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

    fillCache() {
        while (this.cachedText.length < this.cacheSize) {
            let batchIdx = randomBetween(0, this.table.batches.length - 1)

            const text = []

            let rowIdx = null
            for (const obj of this.schema) {
                const column = this.table.batches[batchIdx].getChildAt(obj.idx)
                if (rowIdx === null) {
                    rowIdx = randomBetween(0, column.length - 1)
                    // console.log(
                    //     `has ${this.table.batches.length} batches, with ${
                    //         column.length
                    //     } rows, and ${
                    //         column.length * this.table.batches.length
                    //     } est combinations`
                    // )
                }
                const prefix = obj.value
                text.push(prefix + column.get(rowIdx))
            }

            this.cachedText += text.join(this.delimiter) + this.eosToken
        }
    }

    async getSample(size = 512) {
        this.batches++
        if (this.batches % this.cycleShardInterval === 0) {
            await this.fetchRandomShard()
        }
        this.fillCache()
        const sample = this.cachedText.slice(0, size)
        this.cachedText = this.cachedText.slice(size)
        return sample
    }
}

// async function main() {
//     const sampler = new CosmopediaDataset()
//     await sampler.init()
//     sampler.loadSchema([{ prompt: 'PROMPT: ' }, { text: 'ASSISTANT: ' }])
//     for (let i = 0; i < 10; i++) {
//         console.log(sampler.getSample())
//         console.log('---')
//         console.log('---')
//         console.log('---')
//     }
// }

// main()
