import {layers as $5OpyM$layers, sequential as $5OpyM$sequential, train as $5OpyM$train} from "@tensorflow/tfjs";


async function $c118ee83a19215ed$export$2f453c7f7b0d002b(lstmLayerSize, sampleLen, charSetSize, learningRate) {
    // XXX: Create our processing model. We iterate through the array of lstmLayerSize and
    //      iteratively add an LSTM processing layer whose number of internal units match the
    //      specified value.
    const model = lstmLayerSize.reduce((mdl, lstmLayerSize, i, orig)=>{
        mdl.add($5OpyM$layers.lstm({
            units: lstmLayerSize,
            // XXX: For all layers except the last one, we specify that we'll be returning
            //      sequences of data. This allows us to iteratively chain individual LSTMs
            //      to one-another.
            returnSequences: i < orig.length - 1,
            // XXX: Since each LSTM layer generates a sequence of data, only the first layer
            //      needs to receive a specific input shape. Here, we initialize the inputShape
            //      [sampleLen, charSetSize]. This defines that the first layer will receive an
            //      input matrix which allows us to convert from our selected sample range into
            //      the size of our charset. The charset uses one-hot encoding, which allows us
            //      to represent each possible character in our dataset using a dedicated input
            //      neuron.
            inputShape: i === 0 ? [
                sampleLen,
                charSetSize
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
    model.add($5OpyM$layers.dense({
        units: charSetSize,
        activation: "softmax"
    }));
    // XXX: Finally, compile the model. The optimizer is used to define the backpropagation
    //      technique that should be used for training. We use the rmsProp to help tune the
    //      learning rate that we apply individually to each neuron to help learning.
    //      We use a categoricalCrossentropy loss model which is compatible with our softmax
    //      activation output.
    model.compile({
        optimizer: $5OpyM$train.rmsprop(learningRate),
        loss: "categoricalCrossentropy"
    });
    return model;
}


 // // import '@babel/polyfill'
 // // import '@tensorflow/tfjs-node'
 // // import axios from 'axios'
 // import * as tf from '@tensorflow/tfjs'
 // // XXX: Define the url to pull text from. Here, we're using the tensorflowjs/tfjs Shakespeare
 // //      text corpus (giant blob). What's special about this data source is it shows that a
 // //      neural network can be trained to learn and emulate a specific style of writing. This
 // //      was first popularised by the now famous article "The Unreasonable Effectiveness of
 // //      Neural Networks" by Andrej Kaparthy:
 // //      http://karpathy.github.io/2015/05/21/rnn-effectiveness
 // const url =
 //   'https://storage.googleapis.com/tfjs-examples/lstm-text-generation/data/t8.shakespeare.txt'
 // // XXX: Creates a Long-Short-Term-Memory (LSTM) model. These are specific kinds of Recurrent
 // //      Neural Networks (RNNs) which have optimized neuron coefficients to improve the robustness
 // //      of backpropagation (training). RNNs are effectively chains of layers which process
 // //      individual time steps in a sequence of processing, and exploit the internal state
 // //      (memory) of each layer to achieve complex operations.
 // const createModel = (lstmLayerSize, sampleLen, charSetSize, learningRate) => {
 //   // XXX: Create our processing model. We iterate through the array of lstmLayerSize and
 //   //      iteratively add an LSTM processing layer whose number of internal units match the
 //   //      specified value.
 //   const model = lstmLayerSize.reduce((mdl, lstmLayerSize, i, orig) => {
 //     mdl.add(
 //       tf.layers.lstm({
 //         units: lstmLayerSize,
 //         // XXX: For all layers except the last one, we specify that we'll be returning
 //         //      sequences of data. This allows us to iteratively chain individual LSTMs
 //         //      to one-another.
 //         returnSequences: i < orig.length - 1,
 //         // XXX: Since each LSTM layer generates a sequence of data, only the first layer
 //         //      needs to receive a specific input shape. Here, we initialize the inputShape
 //         //      [sampleLen, charSetSize]. This defines that the first layer will receive an
 //         //      input matrix which allows us to convert from our selected sample range into
 //         //      the size of our charset. The charset uses one-hot encoding, which allows us
 //         //      to represent each possible character in our dataset using a dedicated input
 //         //      neuron.
 //         inputShape: i === 0 ? [sampleLen, charSetSize] : undefined
 //       })
 //     )
 //     // XXX: Here we use a sequential processing model for our network. This model gets passed
 //     //      between each iteration, and is what we add our LSTM layers to.
 //     return mdl
 //   }, tf.sequential())
 //   // XXX: At the output, we use a softmax function (a normalized exponential) as the final
 //   //      classification layer. This is common in many neural networks. It's particularly
 //   //      important for this example, because we use the logit probability model (which
 //   //      supports regression for networks with more than 2 possible outcomes of a categorically
 //   //      distributed dependent variable).
 //   model.add(
 //     tf.layers.dense({
 //       units: charSetSize,
 //       activation: 'softmax'
 //     })
 //   )
 //   // XXX: Finally, compile the model. The optimizer is used to define the backpropagation
 //   //      technique that should be used for training. We use the rmsProp to help tune the
 //   //      learning rate that we apply individually to each neuron to help learning.
 //   //      We use a categoricalCrossentropy loss model which is compatible with our softmax
 //   //      activation output.
 //   model.compile({
 //     optimizer: tf.train.rmsprop(learningRate),
 //     loss: 'categoricalCrossentropy'
 //   })
 //   return model
 // }
 // // XXX: This function separates the input data stream into segments of data
 // //      which can be used for a training epoch. We select segments from the
 // //      input which create the "size" of data we're interested in and use these
 // //      to index from the vectorized character set. These indices need to be
 // //      shuffled to prevent the RNN from accidentally learning the sequence
 // //      these segments usually arrive in. Finally, the mapped data elements
 // //      are packaged into tensors which can be used to drive the inputs and
 // //      outputs of the neural network.
 // const nextDataEpoch = (
 //   textLength,
 //   sampleLen,
 //   sampleStep,
 //   charSetSize,
 //   textIndices,
 //   numExamples
 // ) => {
 //   const trainingIndices = []
 //   for (let i = 0; i < textLength - sampleLen - 1; i += sampleStep) {
 //     trainingIndices.push(i)
 //   }
 //   tf.util.shuffle(trainingIndices)
 //   const xsBuffer = new tf.TensorBuffer([numExamples, sampleLen, charSetSize])
 //   const ysBuffer = new tf.TensorBuffer([numExamples, charSetSize])
 //   for (let i = 0; i < numExamples; ++i) {
 //     const beginIndex = trainingIndices[i % trainingIndices.length]
 //     for (let j = 0; j < sampleLen; ++j) {
 //       xsBuffer.set(1, i, j, textIndices[beginIndex + j])
 //     }
 //     ysBuffer.set(1, i, textIndices[beginIndex + sampleLen])
 //   }
 //   return [xsBuffer.toTensor(), ysBuffer.toTensor()]
 // }
 // // XXX: Takes the generated LSTM character prediction model and uses it
 // //      to predict character data. We use a seed string to initialize the
 // //      generation, which effectively kicks the neural network into producing
 // //      data. Once this window of data is finished, the neural network is
 // //      effectively seeding itself, and "hallucinating" its own output.
 // const generate = (
 //   model,
 //   seed,
 //   sampleLen,
 //   charSetSize,
 //   charSet,
 //   displayLength,
 //   temperature
 // ) => {
 //   // XXX: Fetch the sequence of numeric values which correspond to the
 //   //      sentence.
 //   let sentenceIndices = Array.from(seed).map((e) => charSet.indexOf(e))
 //   let generated = ''
 //   // XXX: Note that since the displayLength is arbitrary, we can make it
 //   //      much larger than our sampleLen. This loop will continue to iterate
 //   //      about the sentenceIndices and buffering the output of the network,
 //   //      which permits it to continue generating far past our initial seed
 //   //      has been provided.
 //   while (generated.length < displayLength) {
 //     const inputBuffer = new tf.TensorBuffer([1, sampleLen, charSetSize])
 //     ;[...Array(sampleLen)].map((_, i) =>
 //       inputBuffer.set(1, 0, i, sentenceIndices[i])
 //     )
 //     const input = inputBuffer.toTensor()
 //     const output = model.predict(input)
 //     // XXX: Pick the character the RNN has decided is the most likely.
 //     //      tf.tidy cleans all of the allocated tensors within the function
 //     //      scope after it has been executed.
 //     const [winnerIndex] = tf.tidy(() =>
 //       // XXX: Draws samples from a multinomial distribution (these are distributions
 //       //      involving multiple variables).
 //       //      tf.squeeze remove dimensions of size (1) from the supplied tensor. These
 //       //      are then divided by the specified temperature.
 //       tf
 //         .multinomial(
 //           // XXX: Use the temperature to control the network's spontaneity.
 //           tf.div(tf.log(tf.squeeze(output)), Math.max(temperature, 1e-6)),
 //           1,
 //           null,
 //           false
 //         )
 //         .dataSync()
 //     )
 //     // XXX: Always clean up tensors once you're finished with them to improve
 //     //      memory utilization and prevent leaks.
 //     input.dispose()
 //     output.dispose()
 //     // XXX: Here we append the generated character to the resulting string, and
 //     //      add this char to the sliding window along the sentenceIndices. This
 //     //      is how we continually wrap around the same buffer and generate arbitrary
 //     //      sequences of data even though our network only accepts fixed inputs.
 //     generated += charSet[winnerIndex]
 //     sentenceIndices = sentenceIndices.slice(1)
 //     sentenceIndices.push(winnerIndex)
 //   }
 //   console.log(`Generated text (temperature=${temperature}):\n ${generated}\n`)
 // }
 // ;(async () => {
 //   // XXX: .
 //   const sampleLen = 60 // length of a sequence of characters we'll pass into the RNN
 //   const sampleStep = 3 // number of characters to jump between segments of input text
 //   const learningRate = 1e-2 // higher values lead to faster convergence, but more errors
 //   const epochs = 150 // the total number of times to update the training weights
 //   const examplesPerEpoch = 10000 // the number of text segments to train against for each epoch
 //   const batchSize = 128 // hyperparameter controlling the frequency weights are updated
 //   const validationSplit = 0.0625 // fraction of training data which will be treated as validation data
 //   const displayLength = 120 // how many characters you want to generate after training
 //   const lstmLayerSize = [128, 128] // the configuration of eah sequential lstm network
 //   const temperatures = [0, 0.25, 0.5, 0.75, 1]
 //   // XXX: Fetch the text data to sample from.
 //   const { data: text } = await axios({
 //     method: 'get',
 //     url
 //   })
 //   // XXX: Fetch all unique characters in the dataset. (quickly!)
 //   const charSet = Array.from(new Set(Array.from(text)))
 //   const { length: charSetSize } = charSet
 //   const model = createModel(lstmLayerSize, sampleLen, charSetSize, learningRate)
 //   model.summary()
 //   // XXX: Convert the total input character text into the corresponding indices in the
 //   //      charSet. This is how we map consistently between character data and numeric
 //   //      neural network dataj
 //   const textIndices = new Uint16Array(
 //     Array.from(text).map((e) => charSet.indexOf(e))
 //   )
 //   // XXX: Pick a random position to start in the dataset. (Note that we choose an index
 //   //      which cannot exceed the minimum size of our sampleLength - 1).
 //   const startIndex = Math.round(Math.random() * (text.length - sampleLen - 1))
 //   // XXX: Create the seed data which we'll use to initialize the network.
 //   const seed = text.slice(startIndex, startIndex + sampleLen)
 //   for (let i = 0; i < epochs; ++i) {
 //     const [xs, ys] = nextDataEpoch(
 //       text.length,
 //       sampleLen,
 //       sampleStep,
 //       charSetSize,
 //       textIndices,
 //       examplesPerEpoch
 //     )
 //     // XXX: Fit the model and hold up iteration of the for loop
 //     //      until it is finished.
 //     await model.fit(xs, ys, {
 //       epochs: 1,
 //       batchSize,
 //       validationSplit,
 //       callbacks: {
 //         onTrainBegin: () => {
 //           console.log(`Epoch ${i + 1} of ${epochs}:`)
 //         },
 //         onTrainEnd: () =>
 //           temperatures.map((temp) =>
 //             generate(
 //               model,
 //               seed,
 //               sampleLen,
 //               charSetSize,
 //               charSet,
 //               displayLength,
 //               temp
 //             )
 //           )
 //       }
 //     })
 //     xs.dispose()
 //     ys.dispose()
 //   }
 // })()


//# sourceMappingURL=module.js.map
