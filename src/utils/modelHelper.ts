import * as ort from 'onnxruntime-web';
import _ from 'lodash';
import { imagenetClasses } from '../data/imagenet';


export async function runSqueezenetModel(preprocessedData: any): Promise<[any, number]> {

  const session = await ort.InferenceSession
    .create('squeezenet1_1.onnx',
      { executionProviders: ['webgl'], graphOptimizationLevel: 'all' });

  var [results, inferenceTime] = await runInference(session, preprocessedData);
  return [results, inferenceTime];
}

async function runInference(session: ort.InferenceSession, preprocessedData: any): Promise<[any, number]> {
  const start = new Date();
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = preprocessedData;

  const outputData = await session.run(feeds);
  const end = new Date();
  const inferenceTime = (end.getTime() - start.getTime()) / 1000;
  const output = outputData[session.outputNames[0]];
  var outputSoftmax = softmax(Array.prototype.slice.call(output.data));

  var results = imagenetClassesTopK(outputSoftmax, 5);
  return [results, inferenceTime];
}

function softmax(resultArray: number[]): any {
  const largestNumber = Math.max(...resultArray);
  const sumOfExp = resultArray.map((resultItem) => Math.exp(resultItem - largestNumber)).reduce((prevNumber, currentNumber) => prevNumber + currentNumber);
  return resultArray.map((resultValue, index) => {
    return Math.exp(resultValue - largestNumber) / sumOfExp;
  });
}

export function imagenetClassesTopK(classProbabilities: any, k = 5) {
  const probs =
    _.isTypedArray(classProbabilities) ? Array.prototype.slice.call(classProbabilities) : classProbabilities;

  const sorted = _.reverse(_.sortBy(probs.map((prob: any, index: number) => [prob, index]), (probIndex: Array<number>) => probIndex[0]));

  const topK = _.take(sorted, k).map((probIndex: Array<number>) => {
    const iClass = imagenetClasses[probIndex[1]];
    return {
      id: iClass[0],
      index: parseInt(probIndex[1].toString(), 10),
      name: iClass[1].replace(/_/g, ' '),
      probability: probIndex[0]
    };
  });
  return topK;
}

