import { Table } from 'apache-arrow'
import { parseRecordBatch } from 'arrow-js-ffi'
import { wasmMemory, readParquetStream } from 'parquet-wasm'

// Large, broken download
const url =
    'https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/data/CC-MAIN-2024-18/001_00003.parquet'

// Small, successful download
// const url =
//     'https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/data/CC-MAIN-2024-18/004_00018.parquet'

;(async () => {
    const stream = await readParquetStream(url)

    const batches = []
    for await (const wasmRecordBatch of stream) {
        const ffiRecordBatch = wasmRecordBatch.intoFFI()
        const recordBatch = parseRecordBatch(
            wasmMemory().buffer,
            ffiRecordBatch.arrayAddr(),
            ffiRecordBatch.schemaAddr()
        )
        batches.push(recordBatch)
        ffiRecordBatch.free()
    }

    const table = new Table(batches)

    console.log('successfully loaded table')
})()
