import * as arrow from 'apache-arrow'
import wasmInit, { readParquet, Table } from 'parquet-wasm'

if (typeof window !== 'undefined') await wasmInit()

export default class CosmopediaDataset {
    constructor(config) {
        this.dataset = 'HuggingFaceTB/cosmopedia'
        this.shard = 'data/stories/train-00000-of-00043.parquet'
    }

    async init() {
        await this.fetchShard()
        this.moveDataIntoTable()
    }

    async fetchShard() {
        const url = `https://huggingface.co/datasets/${this.dataset}/resolve/main/${this.shard}`
        const response = await fetch(url)

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`)
        }

        this.buffer = new Uint8Array(await response.arrayBuffer())
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

    mapSchema(array) {
        this.schema = []
        array.map((obj) => {
            Object.entries(obj).forEach(([key, value]) => {
                console.log(`Key: ${key}`)
                console.log(`Value: ${value}`)
                console.log('---')
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
        console.log(this.schema)
    }

    getPrompt() {
        for (const batch of this.table.batches) {
            const firstColumn = batch.getChildAt(this.schema[0].idx)
            console.log(firstColumn)
            // const promptColumn = batch.getChildAt(promptIndex)
            for (let i = 0; i < firstColumn.length; i++) {
                console.log(`entry ${i}`)
                console.log(firstColumn.get(i))
                // const prompt = promptColumn.get(i)
                // console.log(`Prompt ${count + 1}:`, prompt)
                // count++

                // // Limit the output to first 10 prompts to avoid overwhelming console
                // if (count >= 10) return
                return
            }
        }
        // } else {
        //     console.log("'prompt' field not found in the dataset")
        // }
    }
}

async function main() {
    const sampler = new CosmopediaDataset()
    await sampler.init()
    sampler.mapSchema([{ prompt: 'PROMPT' }, { text: 'ASSISTANT' }])
    sampler.getPrompt()
}

main()
