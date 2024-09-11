import * as arrow from 'apache-arrow'
import { parseTable } from 'arrow-js-ffi'
import { wasmMemory, readParquet } from 'parquet-wasm'

const url =
    'https://huggingface.co/api/datasets/tiiuae/falcon-refinedweb/parquet/default/train/320.parquet'

// This one will succeed
;(async () => {
    const resp = await fetch(url)
    const buffer = new Uint8Array(await resp.arrayBuffer())
    const arrowWasmTable = readParquet(buffer)
    const table = arrow.tableFromIPC(arrowWasmTable.intoIPCStream())

    console.log('successfully loaded table via parquet-wasm')
})()

// This one will fail
;(async () => {
    const resp = await fetch(url)
    const buffer = new Uint8Array(await resp.arrayBuffer())
    const ffiTable = readParquet(buffer).intoFFI()

    const table = parseTable(
        wasmMemory().buffer,
        ffiTable.arrayAddrs(),
        ffiTable.schemaAddr()
    )

    console.log('successfully loaded table via FFI')
})()
