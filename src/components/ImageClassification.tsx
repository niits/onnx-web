import React from "react";
import "bulma/css/bulma.min.css";
import { useRef, useState } from "react";
import * as Jimp from "jimp/browser/lib/jimp";

import { createTensorFromImage } from "../utils/imageHelper";
import { runSqueezenetModel } from "../utils/modelHelper";
interface Props {
  height: number;
  width: number;
}

interface PredictedResult {
  id: string;
  index: number;
  name: string;
  probability: number;
}

const ImageClassification = (props: Props) => {
  const imageRef = useRef<HTMLImageElement>(null);
  const [result, setResult] = useState([] as PredictedResult[]);
  const [hasImage, setHasImage] = useState(false);

  const [inferenceTime, setInferenceTime] = useState(0);

  const handleChange = async (e: React.FormEvent<HTMLInputElement>) => {
    const arrayBuffer = await e.currentTarget.files![0].arrayBuffer();
    const image = await Jimp.read(arrayBuffer as any);

    image.cover(props.height, props.width);

    image.getBase64Async(Jimp.MIME_JPEG).then((newImage) => {
      imageRef.current!.src = newImage;
      setHasImage(true)
    });
  };

  const handleClick = async (e: React.FormEvent<HTMLButtonElement>) => {
    await Jimp.read(imageRef.current!.src).then(async (image: Jimp) => {
      var tensor = createTensorFromImage(image, [
        1,
        3,
        image.bitmap.height,
        image.bitmap.width,
      ]);

      const [predictions, inferenceTime] = await runSqueezenetModel(tensor);
      setResult(predictions);
      setInferenceTime(inferenceTime);
    });
  };
  return (
    <div className="columns is-vcentered">
      <div className="column has-text-centered">
        <div className="box">
          <div className="field is-grouped has-text-centered is-justify-content-center	">
            <div className="control">
              <div className="file">
                <label className="file-label">
                  <input
                    className="file-input"
                    type="file"
                    name="resume"
                    onChange={handleChange}
                  ></input>
                  <span className="file-cta">
                    <span className="file-label">Choose a file…</span>
                  </span>
                </label>
              </div>
            </div>
            <div className="control">
              <button className="button is-link" onClick={handleClick} disabled={!hasImage}>
                Classify
              </button>
            </div>
          </div>
          <figure>
            <img
              ref={imageRef}
              src="https://bulma.io/images/placeholders/256x256.png"
              alt="Input file"
              height={props.height}
              width={props.width}
            />
          </figure>
        </div>
      </div>

      {result.length > 0 ? (
        <div className="column has-text-centered">

          <div className="box">
          <h1>Inference time: {inferenceTime*1000} ms</h1>
            {result.map(function (object: PredictedResult, i) {
              return (
                <div key={i} className="columns is-vcentered is-multiline">
                  <div className="column is-one-quarter">
                    <b>{object.name}</b>
                  </div>
                  <div className="column is-three-quarters">
                    <progress
                      className="progress is-primary"
                      value={object.probability * 100}
                      max="100"
                    >
                      {object.probability * 100}%
                    </progress>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ) : (
        ""
      )}
    </div>
  );
};

export default ImageClassification;
