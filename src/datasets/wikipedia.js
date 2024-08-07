import { parseTable } from 'arrow-js-ffi'
import initWasm, { wasmMemory, readParquet } from 'parquet-wasm'
import { randomBetween, randomValueFromArray } from '../utils.js'

export default class FinewebDataset {
    constructor(config) {
        this.dataset = 'wikimedia/wikipedia'
        this.slices = [{ slice: '20231101.en', shards: 41 }]
        this.split = 'train'
        this.delimiter = '\n\n'
        this.eosToken = config?.eosToken || '֍'
        this.batchesBeforeRefresh = config?.batchesBeforeRefresh || 10000
        this.batches = 0
        this.cacheSize = 20000
        this.cachedText = ''
        this.table = {}
        this.schemaTemplate = config?.schema
    }

    async init() {
        const isBrowser =
            (typeof self !== 'undefined' &&
                typeof self.importScripts === 'function') ||
            typeof window !== 'undefined'
        if (isBrowser) await initWasm()
        await this.fetchRandomShard()
    }

    generatePaddedNumbers(split, numShards) {
        const numbers = []
        const suffix = String(numShards).padStart(5, '0')
        for (let i = 0; i < numShards; i++) {
            const prefix = String(i).padStart(5, '0')
            numbers.push(`${split}-${prefix}-of-${suffix}`)
        }
        return numbers
    }

    async fetchRandomShard() {
        const { slice, shards } = randomValueFromArray(this.slices)
        const shardIndices = this.generatePaddedNumbers(this.split, shards)
        const shard = randomValueFromArray(shardIndices)
        const path = `${slice}/${shard}.parquet`
        const url = `https://huggingface.co/datasets/${this.dataset}/resolve/main/${path}`
        console.log(
            'fetching dataset:',
            this.dataset,
            'shard:',
            `${shard}`,
            'slice:',
            slice
        )
        try {
            await this.streamDataIntoTable(url)
        } catch (err) {
            console.error(err)
            console.warn(
                `Failed to fetch shard (${shard}) from HuggingFace! We will continue using the old one for now...`
            )
        }
        this.loadSchema(this.schemaTemplate)
    }

    async streamDataIntoTable(url) {
        const resp = await fetch(url)
        const buffer = new Uint8Array(await resp.arrayBuffer())
        const ffiTable = readParquet(buffer).intoFFI()

        this.table = parseTable(
            wasmMemory().buffer,
            ffiTable.arrayAddrs(),
            ffiTable.schemaAddr()
        )

        ffiTable.drop()
    }

    loadSchema(array = [{ title: 'INPUT: ', text: 'OUTPUT: ' }]) {
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
            let batchIdx = randomBetween(0, this.table.batches.length - 1)

            const text = []

            let rowIdx = null
            for (const field of this.schema) {
                let column = this.table.batches[batchIdx].getChildAt(field.idx)
                if (rowIdx === null) {
                    rowIdx = randomBetween(0, column.length - 1)
                }
                const prefix = field.value
                const data = column.get(rowIdx)
                text.push(prefix + data)
            }
            this.cachedText += text.join(this.delimiter) + this.eosToken
        }
    }

    async getSample(size = 512) {
        this.batches++
        try {
            if (this.batches % this.batchesBeforeRefresh === 0) {
                await this.fetchRandomShard()
            }
            await this.fillCache()
            const sample = this.cachedText.slice(0, size)
            this.cachedText = this.cachedText.slice(size)
            return sample
        } catch (err) {
            console.error(err)
            return await this.getSample(size)
        }
    }
}

// async function main() {
//     const sampler = new FinewebDataset({
//         schema: [{ title: 'INPUT: ', text: 'OUTPUT: ' }]
//     })
//     await sampler.init()
//     for (let i = 0; i < 10; i++) {
//         console.log(await sampler.getSample())
//         console.log('---')
//         console.log('---')
//         console.log('---')
//     }
// }

// main()
