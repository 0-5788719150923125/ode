import {backend as $5OpyM$backend, layers as $5OpyM$layers, sequential as $5OpyM$sequential, train as $5OpyM$train, TensorBuffer as $5OpyM$TensorBuffer, tidy as $5OpyM$tidy, multinomial as $5OpyM$multinomial, div as $5OpyM$div, log as $5OpyM$log, squeeze as $5OpyM$squeeze, data as $5OpyM$data, buffer as $5OpyM$buffer} from "@tensorflow/tfjs-node-gpu";

// import '@tensorflow/tfjs-node'


function $b01b19e983ceb86f$export$4dc0b9eca0839ce2(min, max) {
    return Math.floor(Math.random() * (max - min + 1) + min);
}


async function $1a004f8cf919e722$export$9bc55a205916dc35(dataGenerator, batchSize = 64) {
    const validationSplit = 0.0625 // fraction of training data which will be treated as validation data
    ;
    const seed = "";
    const ds = $5OpyM$data.generator($1a004f8cf919e722$var$createBatchGenerator(dataGenerator, this.vocab));
    await this.model.fitDataset(ds, {
        epochs: 1000,
        batchSize: batchSize,
        // validationSplit,
        callbacks: {
            onTrainBegin: ()=>{},
            onBatchEnd: async (batch, logs)=>{
                if (batch % 128 === 0) {
                    console.log(logs);
                    for(let temp in [
                        0,
                        0.3,
                        0.7,
                        0.9,
                        1.1
                    ]){
                        const output = await this.generate(seed, temp);
                        console.log(output);
                    }
                }
            },
            onEpochEnd: async (epoch, logs)=>console.log("epoch ended")
        }
    });
}
function $1a004f8cf919e722$var$createBatchGenerator(dataGenerator, vocab) {
    return function*() {
        yield* $1a004f8cf919e722$var$batchGenerator(dataGenerator, vocab);
    };
}
function* $1a004f8cf919e722$var$batchGenerator(dataGenerator, vocab) {
    while(true){
        const text = dataGenerator.next().value;
        console.log(text);
        // Extract necessary parameters directly
        const filteredText = text.split("").filter((e)=>vocab.indexOf(e) !== -1).join("");
        const textIndices = new Uint16Array(filteredText.split("").map((e)=>vocab.indexOf(e)));
        const sampleLength = textIndices.length - 1;
        // Create tensors directly for the single batch
        const xsBuffer = $5OpyM$buffer([
            1,
            sampleLength,
            vocab.length
        ]);
        const ysBuffer = $5OpyM$buffer([
            1,
            vocab.length
        ]);
        // Fill the tensors directly without intermediate arrays
        for(let i = 0; i < sampleLength; ++i)xsBuffer.set(1, 0, i, textIndices[i]);
        ysBuffer.set(1, 0, textIndices[sampleLength]);
        yield {
            xs: xsBuffer.toTensor(),
            ys: ysBuffer.toTensor()
        };
    }
}


console.log("Backend:", $5OpyM$backend());
class $b8c69f6a386226b6$export$2e2bcd8739ae039 {
    constructor(lstmLayerSize, sampleLen, learningRate, displayLength){
        this.lstmLayerSize = lstmLayerSize;
        this.sampleLen = sampleLen;
        this.vocab = Array.from(new Set(Array.from(`\xb60123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?!'"(){}[]|/\\
 `)));
        this.learningRate = learningRate;
        this.displayLength = displayLength;
        this.model = null;
        this.init();
    }
    init() {
        // XXX: Create our processing model. We iterate through the array of lstmLayerSize and
        //      iteratively add an LSTM processing layer whose number of internal units match the
        //      specified value.
        this.model = this.lstmLayerSize.reduce((mdl, lstmLayerSize, i, orig)=>{
            mdl.add($5OpyM$layers.lstm({
                units: lstmLayerSize,
                // XXX: For all layers except the last one, we specify that we'll be returning
                //      sequences of data. This allows us to iteratively chain individual LSTMs
                //      to one-another.
                returnSequences: i < orig.length - 1,
                // XXX: Since each LSTM layer generates a sequence of data, only the first layer
                //      needs to receive a specific input shape. Here, we initialize the inputShape
                //      [sampleLen, this.vocab.length]. This defines that the first layer will receive an
                //      input matrix which allows us to convert from our selected sample range into
                //      the size of our charset. The charset uses one-hot encoding, which allows us
                //      to represent each possible character in our dataset using a dedicated input
                //      neuron.
                inputShape: i === 0 ? [
                    this.sampleLen,
                    this.vocab.length
                ] : undefined
            }));
            // XXX: Here we use a sequential processing model for our network. This model gets passed
            //      between each iteration, and is what we add our LSTM layers to.
            return mdl;
        }, $5OpyM$sequential());
        // XXX: At the output, we use a softmax function (a normalized exponential) as the final
        //      classification layer. This is common in many neural networks. It's particularly
        //      important for this example, because we use the logit probability model (which
        //      supports regression for networks with more than 2 possible outcomes of a categorically
        //      distributed dependent variable).
        this.model.add($5OpyM$layers.dense({
            units: this.vocab.length,
            activation: "softmax"
        }));
        // XXX: Finally, compile the model. The optimizer is used to define the backpropagation
        //      technique that should be used for training. We use the rmsProp to help tune the
        //      learning rate that we apply individually to each neuron to help learning.
        //      We use a categoricalCrossentropy loss model which is compatible with our softmax
        //      activation output.
        this.model.compile({
            optimizer: $5OpyM$train.rmsprop(this.learningRate),
            loss: "categoricalCrossentropy"
        });
    }
    getModel() {
        return this.model;
    }
    async trainModel(dataGenerator) {
        const bound = (0, $1a004f8cf919e722$export$9bc55a205916dc35).bind(this);
        await bound(dataGenerator);
    }
    getWeights() {
        return this.model.getWeights();
    }
    async generate(seed, temperature = 0.7) {
        const bound = $b8c69f6a386226b6$var$generate.bind(this);
        return await bound(seed, temperature);
    }
}
async function $b8c69f6a386226b6$var$generate(seed, temperature) {
    // XXX: Fetch the sequence of numeric values which correspond to the
    //      sentence.
    let sentenceIndices = Array.from(seed).map((e)=>this.vocab.indexOf(e));
    let generated = "";
    // XXX: Note that since the displayLength is arbitrary, we can make it
    //      much larger than our sampleLen. This loop will continue to iterate
    //      about the sentenceIndices and buffering the output of the network,
    //      which permits it to continue generating far past our initial seed
    //      has been provided.
    while(generated.length < this.displayLength){
        const inputBuffer = new $5OpyM$TensorBuffer([
            1,
            this.sampleLen,
            this.vocab.length
        ]);
        [
            ...Array(this.sampleLen)
        ].map((_, i)=>inputBuffer.set(1, 0, i, sentenceIndices[i]));
        const input = inputBuffer.toTensor();
        const output = this.model.predict(input);
        // XXX: Pick the character the RNN has decided is the most likely.
        //      tf.tidy cleans all of the allocated tensors within the function
        //      scope after it has been executed.
        const [winnerIndex] = $5OpyM$tidy(()=>// XXX: Draws samples from a multinomial distribution (these are distributions
            //      involving multiple variables).
            //      tf.squeeze remove dimensions of size (1) from the supplied tensor. These
            //      are then divided by the specified temperature.
            $5OpyM$multinomial(// XXX: Use the temperature to control the network's spontaneity.
            $5OpyM$div($5OpyM$log($5OpyM$squeeze(output)), Math.max(temperature, 1e-6)), 1, null, false).dataSync());
        // XXX: Always clean up tensors once you're finished with them to improve
        //      memory utilization and prevent leaks.
        input.dispose();
        output.dispose();
        // XXX: Here we append the generated character to the resulting string, and
        //      add this char to the sliding window along the sentenceIndices. This
        //      is how we continually wrap around the same buffer and generate arbitrary
        //      sequences of data even though our network only accepts fixed inputs.
        generated += this.vocab[winnerIndex];
        sentenceIndices = sentenceIndices.slice(1);
        sentenceIndices.push(winnerIndex);
    }
    // console.log(`Generated text (temperature=${temperature}):\n ${generated}\n`)
    return generated;
}


var $cf838c15c8b009ba$export$2e2bcd8739ae039 = (0, $b8c69f6a386226b6$export$2e2bcd8739ae039);


export {$cf838c15c8b009ba$export$2e2bcd8739ae039 as default};
//# sourceMappingURL=index.js.map
