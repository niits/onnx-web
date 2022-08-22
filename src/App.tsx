import React from "react";
import "bulma/css/bulma.min.css";

import ImageClassification from "./components/ImageClassification";

function App() {
  return (
    <section className="hero">
      <div className="hero-body">
        <p className="title">ONNX Runtime Web Demo</p>
        <p className="subtitle">Classify images with SqueezeNet</p>
      </div>
      <div className="container  is-fluid">
        <ImageClassification width={224} height={224} />
        <div id="result" className="mt-3">
          {" "}
        </div>
      </div>
    </section>
  );
}

export default App;
