function Spinner({ config }) {
  const ref = React.useRef(null);

  React.useEffect(() => {
    const spinner = new Spin.Spinner({
      lines: 13,
      color: "#ffffff",
      ...config,
    });
    spinner.spin(ref.current);
    return () => spinner.stop();
  }, [ref]);

  return <span ref={ref} />;
}

function Result({ callId, selectedFile }) {
  const [result, setResult] = React.useState();
  const [intervalId, setIntervalId] = React.useState();

  React.useEffect(() => {
    if (result) {
      clearInterval(intervalId);
      return;
    }

    const _intervalID = setInterval(async () => {
      const resp = await fetch(`/result/${callId}`);
      if (resp.status === 200) {
        setResult(await resp.json());
      }
    }, 100);

    setIntervalId(_intervalID);

    return () => clearInterval(intervalId);
  }, [result]);

  return (
    <div class="flex items-center content-center justify-center space-x-4 ">
      <img src={URL.createObjectURL(selectedFile)} class="h-[300px]" />
      {!result && <Spinner config={{}} />}
      {result && (
        <p class="w-[500px] p-4 bg-zinc-200 rounded-lg whitespace-pre-wrap text-xs font-mono">
          {JSON.stringify(result[0], undefined, 1)}
        </p>
      )}
    </div>
  );
}

function Form({ onSubmit, onFileSelect, selectedFile, maxLength, onMaxLengthChange, numBeams, onNumBeamsChange }) {
  return (
    <form class="flex flex-col space-y-4 items-center">
      <div class="text-2xl font-semibold text-gray-700"> ViT-GPT2 Image Captioning </div>
      <input
        accept="image/*"
        type="file"
        name="file"
        onChange={onFileSelect}
        class="block w-full text-sm text-gray-900 bg-gray-50 rounded-lg border border-gray-300 cursor-pointer"
      />
      {selectedFile ? (
        <img src={URL.createObjectURL(selectedFile)} class="h-[300px]" />
      ) : null}
      <div className="flex justify-center items-center space-x-4">
        <label htmlFor="max-length-slider">Max Length: {maxLength}</label>
        <input
          type="range"
          id="max-length-slider"
          name="max_length"
          min="10"
          max="500"
          value={maxLength}
          onChange={onMaxLengthChange}
        />
      </div>
      <div className="flex justify-center items-center space-x-4">
        <label htmlFor="num-beams-slider">Num Beams: {numBeams}</label>
        <input
          type="range"
          id="num-beams-slider"
          name="num_beams"
          min="1"
          max="20"
          value={numBeams}
          onChange={onNumBeamsChange}
        />
      </div>
      <div>
        <button
          type="button"
          onClick={onSubmit}
          disabled={!selectedFile}
          class="bg-indigo-400 disabled:bg-zinc-500 hover:bg-indigo-600 text-white font-bold py-2 px-4 rounded text-sm"
        >
          Upload
        </button>
      </div>
    </form>
  );
}

function App() {
  const [selectedFile, setSelectedFile] = React.useState();
  const [callId, setCallId] = React.useState();
  const [maxLength, setMaxLength] = React.useState(50);
  const [numBeams, setNumBeams] = React.useState(4);

  const handleSubmission = async () => {
    const formData = new FormData();
    formData.append("image", selectedFile);
    formData.append("max_length", maxLength);
    formData.append("num_beams", numBeams);

    const resp = await fetch("/parse", {
      method: "POST",
      body: formData,
    });

    if (resp.status !== 200) {
      throw new Error("An error occurred: " + resp.status);
    }
    const body = await resp.json();
    setCallId(body.call_id);
  };

  return (
    <div class="absolute inset-0 bg-gradient-to-r from-indigo-300 via-purple-300 to-pink-300">
      <div class="mx-auto max-w-md py-8">
        <main class="rounded-xl bg-white p-6">
          {!callId && (
            <Form
              onSubmit={handleSubmission}
              onFileSelect={(e) => setSelectedFile(e.target.files[0])}
              selectedFile={selectedFile}
              onMaxLengthChange={(e) => setMaxLength(e.target.value)}
              maxLength={maxLength}
              onNumBeamsChange={(e) => setNumBeams(e.target.value)}
              numBeams={numBeams}
            />
          )}
          {callId && <Result callId={callId} selectedFile={selectedFile} maxLength={maxLength} numBeams={numBeams} />}
        </main>
      </div>
    </div>
  );
}

const container = document.getElementById("react");
ReactDOM.createRoot(container).render(<App />);
