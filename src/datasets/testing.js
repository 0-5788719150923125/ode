import { Table, tableFromIPC } from 'apache-arrow'
import { readParquetStream } from 'parquet-wasm'

const stream = await readParquetStream(
    'https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/resolve/main/data/web_samples_v2/train-00005-of-00118.parquet'
)

const batches = []
for await (const wasmRecordBatch of stream) {
    console.log(Math.random())
    const arrowTable = tableFromIPC(wasmRecordBatch.intoIPCStream())
    batches.push(...arrowTable.batches)
}
const table = new Table(batches)
console.log(table)
