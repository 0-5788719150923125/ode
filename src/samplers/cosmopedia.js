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
        const { slice, shards } = randomValueFromArray(this.slices)
        const allShards = generatePaddedNumbers(0, shards, 5)
        let shard = randomValueFromArray(allShards.slice(0, -2))
        if (typeof shard === 'undefined') shard = '00000'
        console.log('fetching shard:', shard, 'slice:', slice)
        const path = `data/${slice}/${this.split}-${shard}-of-${allShards.slice(
            -1
        )}.parquet`
        const url = `https://huggingface.co/datasets/${this.dataset}/resolve/main/${path}`
        console.log(url)
        const response = await fetch(url)
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`)
        }

        this.buffer = new Uint8Array(await response.arrayBuffer())
        this.moveDataIntoTable()
        console.log('moved shard to table:', shard)
    }

    moveDataIntoTable() {
        // Read Parquet buffer to Arrow Table
        const arrowWasmTable = readParquet(this.buffer)
        // Convert to JS Arrow Table
        this.table = arrow.tableFromIPC(arrowWasmTable.intoIPCStream())
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
            let idx = 0
            let batchIdx = 0
            let rowIdx = 0
            try {
                batchIdx = randomBetween(0, this.table.batches.length - 1)

                const text = []

                rowIdx = null
                for (const obj of this.schema) {
                    idx = obj.idx
                    const column = this.table.batches[batchIdx].getChildAt(
                        obj.idx
                    )
                    if (rowIdx === null) {
                        rowIdx = randomBetween(0, column.length - 1)
                    }
                    const prefix = obj.value
                    text.push(prefix + column.get(rowIdx))
                }

                this.cachedText += text.join(this.delimiter) + this.eosToken
            } catch (err) {
                console.error(err)
                console.log('idx was:', idx)
                console.log('batch idx was:', batchIdx)
                console.log('batch object is:', this.table.batches[batchIdx])
                console.log('row idx was:', rowIdx)
            }
        }
    }

    async getSample(size = 512) {
        this.batches++
        if (this.batches % this.cycleShardInterval === 0) {
            await this.fetchRandomShard()
        }
        this.fillCache()
        const sample = this.cachedText.slice(0, size)
        this.cachedText = this.cachedText.slice(size, -1)
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
