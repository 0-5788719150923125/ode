import "@tensorflow/tfjs-node";
import {layers as $5OpyM$layers, sequential as $5OpyM$sequential, train as $5OpyM$train, TensorBuffer as $5OpyM$TensorBuffer, tidy as $5OpyM$tidy, multinomial as $5OpyM$multinomial, div as $5OpyM$div, log as $5OpyM$log, squeeze as $5OpyM$squeeze, data as $5OpyM$data, util as $5OpyM$util} from "@tensorflow/tfjs";





async function $1a004f8cf919e722$export$9bc55a205916dc35(dataGenerator) {
    // XXX: .
    const sampleLen = 60 // length of a sequence of characters we'll pass into the RNN
    ;
    const sampleStep = 3 // number of characters to jump between segments of input text
    ;
    const epochs = 150 // the total number of times to update the training weights
    ;
    const examplesPerEpoch = 10000 // the number of text segments to train against for each epoch
    ;
    const batchSize = 128 // hyperparameter controlling the frequency weights are updated
    ;
    const validationSplit = 0.0625 // fraction of training data which will be treated as validation data
    ;
    const seed = "";
    for(let i = 0; i < epochs; ++i){
        const ds = $5OpyM$data.generator($1a004f8cf919e722$var$createBatchGenerator(dataGenerator, self.characters));
        await this.model.fitDataset(ds, {
            epochs: 1,
            batchSize: batchSize,
            // validationSplit,
            callbacks: {
                onTrainBegin: ()=>console.log(`Epoch ${i + 1} of ${epochs}:`),
                onBatchEnd: async (batch, logs)=>console.log(logs),
                onEpochEnd: async (epoch, logs)=>console.log(await this.generate(seed, 0.7))
            }
        });
    }
}
function $1a004f8cf919e722$var$createBatchGenerator(dataGenerator) {
    return function*() {
        yield* $1a004f8cf919e722$var$batchGenerator(dataGenerator, characters);
    };
}
function* $1a004f8cf919e722$var$batchGenerator(dataGenerator, characters1) {
    console.log("trying to load batches");
    while(true){
        const text = dataGenerator.next().value;
        console.log(text);
        // Extract necessary parameters from text or context
        const textLength = text.length;
        const sampleLen = 60 // Adjust as needed
        ;
        const sampleStep = 3 // Adjust as needed
        ;
        // const charSet = Array.from(new Set(Array.from(text)))
        // const charSetSize = charSet.length
        // Create tensors for the current batch
        const textIndices = new Uint16Array(Array.from(text).map((e)=>characters1.indexOf(e)));
        const trainingIndices = [];
        for(let i = 0; i < textLength - sampleLen - 1; i += sampleStep)trainingIndices.push(i);
        $5OpyM$util.shuffle(trainingIndices);
        const xsBuffer = new $5OpyM$TensorBuffer([
            1,
            sampleLen,
            characters1.length
        ]) // One example per batch
        ;
        const ysBuffer = new $5OpyM$TensorBuffer([
            1,
            characters1.length
        ]);
        const batchIndex = trainingIndices[0 % trainingIndices.length];
        for(let j = 0; j < sampleLen; ++j)xsBuffer.set(1, 0, j, textIndices[batchIndex + j]);
        ysBuffer.set(1, 0, textIndices[batchIndex + sampleLen]);
        yield {
            xs: xsBuffer.toTensor(),
            ys: ysBuffer.toTensor()
        };
    }
}


class $b8c69f6a386226b6$export$2e2bcd8739ae039 {
    constructor(lstmLayerSize, sampleLen, learningRate, displayLength){
        this.lstmLayerSize = lstmLayerSize;
        this.sampleLen = sampleLen;
        this.characters = Array.from(new Set(Array.from(`\xb60123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?!'"(){}[]| `)));
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
                //      [sampleLen, this.characters.length]. This defines that the first layer will receive an
                //      input matrix which allows us to convert from our selected sample range into
                //      the size of our charset. The charset uses one-hot encoding, which allows us
                //      to represent each possible character in our dataset using a dedicated input
                //      neuron.
                inputShape: i === 0 ? [
                    this.sampleLen,
                    this.characters.length
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
            units: this.characters.length,
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
    let sentenceIndices = Array.from(seed).map((e)=>this.characters.indexOf(e));
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
            this.characters.length
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
        generated += this.characters[winnerIndex];
        sentenceIndices = sentenceIndices.slice(1);
        sentenceIndices.push(winnerIndex);
    }
    // console.log(`Generated text (temperature=${temperature}):\n ${generated}\n`)
    return generated;
}


var $cf838c15c8b009ba$export$2e2bcd8739ae039 = (0, $b8c69f6a386226b6$export$2e2bcd8739ae039);


export {$cf838c15c8b009ba$export$2e2bcd8739ae039 as default};
//# sourceMappingURL=index.js.map
