import * as arrow from 'apache-arrow'
import wasmInit, { readParquet, Table } from 'parquet-wasm'
// import fetch from 'node-fetch'

// // Instantiate the WebAssembly context
// await wasmInit()

async function fetchParquetFromHuggingFace(dataset, file) {
    const url = `https://huggingface.co/datasets/${dataset}/resolve/main/${file}`
    const response = await fetch(url)

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
    }

    return new Uint8Array(await response.arrayBuffer())
}

async function processParquetFile(parquetUint8Array) {
    // Read Parquet buffer to Arrow Table
    const arrowWasmTable = readParquet(parquetUint8Array)

    // Convert to JS Arrow Table
    const table = arrow.tableFromIPC(arrowWasmTable.intoIPCStream())

    console.log(table.schema.toString())

    // Get the 'prompt' field index
    const promptIndex = table.schema.fields.findIndex(
        (field) => field.name === 'prompt'
    )

    if (promptIndex !== -1) {
        // Iterate over the batches and extract prompts
        let count = 0
        for (const batch of table.batches) {
            const promptColumn = batch.getChildAt(promptIndex)
            for (let i = 0; i < promptColumn.length; i++) {
                const prompt = promptColumn.get(i)
                console.log(`Prompt ${count + 1}:`, prompt)
                count++

                // Limit the output to first 10 prompts to avoid overwhelming console
                if (count >= 10) return
            }
        }
    } else {
        console.log("'prompt' field not found in the dataset")
    }
}

async function main() {
    const dataset = 'HuggingFaceTB/cosmopedia'
    const file = 'data/stories/train-00000-of-00043.parquet'

    try {
        const parquetData = await fetchParquetFromHuggingFace(dataset, file)
        await processParquetFile(parquetData)
    } catch (error) {
        console.error('Error processing the Parquet file:', error)
    }
}

main()
